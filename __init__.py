"""ComfyUI integration for the isolated local Drama 3 FP8 runtime."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import folder_paths
import comfy.model_management as model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io
from server import PromptServer
from .take_io import load_take, save_take

CONFIG = Path(__file__).resolve().parent / 'runtime.json'
DEFAULT_RUNTIME = Path(os.environ.get('DRAMA3_RUNTIME') or
    (json.loads(CONFIG.read_text(encoding='utf-8-sig'))['runtime_folder'] if CONFIG.is_file() else r'D:\Models\Drama3Runtime'))
VOICE = io.Custom('DRAMA3_VOICE')
VOICES = io.Custom('DRAMA3_VOICES')


def write_audio(path, audio):
    waveform = audio['waveform']
    if waveform.shape[0] != 1:
        raise ValueError('Connect a single audio clip per reference.')
    samples = waveform[0].detach().cpu().float().T.numpy()
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('Audio must contain finite samples.')
    sf.write(path, samples, audio['sample_rate'], subtype='FLOAT')


class Drama3FP8Speech:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'text': ('STRING', {'multiline': True, 'default': 'Hello, this is a local test.'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff}),
            'max_frames': ('INT', {'default': 0, 'min': 0, 'max': 16384, 'step': 32,
                'tooltip': 'Per part: 0 = automatic (600 frames for long scripts, 2048 for a single part). 256 frames is about 12 seconds.'}),
            'runtime_folder': ('STRING', {'default': '', 'tooltip': 'Leave blank to use the installed runtime configuration.'}),
            'release_comfy_models': ('BOOLEAN', {'default': True}),
        }, 'optional': {
            'reference_audio': ('AUDIO',),
            'reference_text': ('STRING', {'default': '', 'multiline': True, 'tooltip': 'Exact words spoken in reference_audio (speaker 0).'}),
            'temperature': ('FLOAT', {'default': .7, 'min': .01, 'max': 2., 'step': .05}),
            'top_p': ('FLOAT', {'default': .7, 'min': .01, 'max': 1., 'step': .05}),
            'top_k': ('INT', {'default': 30, 'min': 1, 'max': 4096}),
            'voices': ('DRAMA3_VOICES',),
            'previous_take': ('DRAMA3_TAKE',),
            'direction': ('STRING', {'default': '', 'multiline': True, 'tooltip': 'Natural language delivery instruction, e.g. warm, amused, speaking softly. Applied to each speaker.'}),
            'chunk_mode': (['auto', 'single', 'lines', 'clauses'], {'default': 'auto',
                'tooltip': 'clauses generates each comma-separated clause separately, preventing an early stop from dropping subsequent clauses.'}),
            'chunk_bytes': ('INT', {'default': 300, 'min': 50, 'max': 8192, 'step': 50}),
            'history_turns': ('INT', {'default': 1, 'min': 0, 'max': 100, 'tooltip': 'Recent turns kept in context, in addition to the first turn. Older history is pruned when the prompt is full.'}),
            'use_history': ('BOOLEAN', {'default': True, 'tooltip': 'Use previous audio as context. Disable to retain its reference voices but start fresh.'}),
            'variations': ('INT', {'default': 1, 'min': 1, 'max': 64, 'tooltip': 'Seeds increment across takes. Model is loaded once per job.'}),
            'streaming': ('BOOLEAN', {'default': False, 'tooltip': 'Decode and preview during generation. Uses more RAM offloading and may run slower.'}),
        }, 'hidden': {'unique_id': 'UNIQUE_ID'}}

    RETURN_TYPES = ('AUDIO', 'DRAMA3_TAKE', 'STRING')
    RETURN_NAMES = ('audio', 'take', 'generation_info')
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = 'synthesize'
    CATEGORY = 'audio/Drama 3'
    DESCRIPTION = 'Original FP8 local speech, voice cloning, speaker-tagged dialogue, long scripts, previous-take context, variations and progressive preview.'

    def synthesize(self, text, seed, max_frames, runtime_folder, release_comfy_models,
                   reference_audio=None, reference_text='', temperature=.7, top_p=.7, top_k=30,
                   voices=None, previous_take=None, direction='', chunk_mode='auto', chunk_bytes=300,
                   history_turns=1, use_history=True, variations=1, streaming=False, unique_id=None):
        if not text.strip():
            raise ValueError('Enter some text to speak.')
        if reference_audio is not None and not reference_text.strip():
            raise ValueError('Enter the exact reference transcript in reference_text.')
        if reference_audio is None and reference_text.strip():
            raise ValueError('Connect reference_audio, or clear reference_text.')
        runtime = Path(runtime_folder.strip()).expanduser().resolve() if runtime_folder.strip() else DEFAULT_RUNTIME.resolve()
        python = runtime / 'drama3-env/Scripts/python.exe'
        runner = runtime / 'run_drama3_full.py'
        if not python.is_file() or not runner.is_file():
            raise RuntimeError(f'The complete Drama 3 runtime is missing from {runtime}.')
        if release_comfy_models:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(0)
        budget = min(26., free_bytes / 2**30 - 2.5)
        if budget < (11.5 if streaming else 8.):
            raise RuntimeError('Free other GPU workloads before running Drama 3.')
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        session = uuid.uuid4().hex
        client_id = PromptServer.instance.client_id
        progress = ProgressBar(1000)

        def notify(event):
            if 'progress' in event:
                progress.update_absolute(round(event['progress'] * 1000))
            if client_id is not None:
                PromptServer.instance.send_sync('drama3.preview', event | {'node_id': str(unique_id), 'session': session}, sid=client_id)

        with tempfile.TemporaryDirectory(prefix='drama3_', dir=str(temp_root), ignore_cleanup_errors=True) as temp:
            temp = Path(temp)
            output = temp / 'speech.wav'
            job = {'text': text, 'seed': seed, 'max_frames': max_frames, 'temperature': temperature,
                   'top_p': top_p, 'top_k': top_k, 'direction': direction, 'chunk_mode': chunk_mode,
                   'chunk_bytes': chunk_bytes, 'history_turns': history_turns, 'use_history': use_history,
                   'variations': variations, 'streaming': streaming, 'voices': []}
            references = list(voices or [])
            if reference_audio is not None:
                references.append({'audio': reference_audio, 'text': reference_text, 'speaker_id': 0})
            for index, voice in enumerate(references):
                path = temp / f'reference_{index}.wav'
                write_audio(path, voice['audio'])
                job['voices'].append({'path': str(path), 'text': voice['text'], 'speaker_id': voice['speaker_id']})
            if previous_take is not None:
                if 'audio' in previous_take:
                    path = temp / 'previous.wav'
                    write_audio(path, previous_take['audio'])
                    job.update(previous_audio=str(path), previous_text=previous_take['text'])
                else:
                    path = temp / 'previous.npz'
                    save_take(path, previous_take)
                    job['previous_take'] = str(path)
            request = temp / 'request.json'
            request.write_text(json.dumps(job), encoding='utf-8')
            command = [str(python), '-u', str(runner), '--request', str(request), '--gpu-budget', str(budget), '--output', str(output)]
            log_path = temp / 'generation.log'
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            offset = 0

            def read_events():
                nonlocal offset
                events = temp / 'events.jsonl'
                if not events.exists():
                    return
                with events.open('r', encoding='utf-8') as file:
                    file.seek(offset)
                    while line := file.readline():
                        if not line.endswith('\n'):
                            break
                        offset = file.tell()
                        event = json.loads(line)
                        if 'audio' in event:
                            source = (temp / event['audio']).resolve()
                            if source.parent != temp.resolve():
                                raise ValueError('Invalid preview path.')
                            name = f'drama3_preview_{session}_{source.name}'
                            shutil.copyfile(source, temp_root / name)
                            event.update(audio=name, type='temp', subfolder='')
                        notify(event)

            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, cwd=str(runtime))
                try:
                    while process.poll() is None:
                        model_management.throw_exception_if_processing_interrupted()
                        read_events()
                        time.sleep(.25)
                    read_events()
                except BaseException:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    notify({'stage': 'Stopped'})
                    raise
            if process.returncode or not output.is_file():
                detail = log_path.read_text(encoding='utf-8', errors='replace')[-6000:]
                notify({'stage': 'Failed - see error details'})
                raise RuntimeError(f'Drama 3 generation failed:\n{detail}')
            stats = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            audios, takes, infos = [], [], []
            for result in stats['results']:
                audio_path = (temp / result['audio']).resolve()
                take_path = (temp / result['take']).resolve()
                if audio_path.parent != temp.resolve() or take_path.parent != temp.resolve():
                    raise ValueError('Invalid runtime result path.')
                samples, rate = sf.read(audio_path, dtype='float32', always_2d=True)
                audios.append({'waveform': torch.from_numpy(np.ascontiguousarray(samples.T)).unsqueeze(0), 'sample_rate': rate})
                takes.append(load_take(take_path))
                infos.append(json.dumps({k: v for k, v in stats.items() if k != 'results'} | result, indent=2))
            progress.update_absolute(1000)
            return audios, takes, infos


class Drama3Voice:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': ('AUDIO',), 'transcript': ('STRING', {'default': '', 'multiline': True}),
                             'speaker_id': ('INT', {'default': 0, 'min': 0, 'max': 15})}}
    RETURN_TYPES = ('DRAMA3_VOICE',)
    FUNCTION = 'create'
    CATEGORY = 'audio/Drama 3'

    def create(self, audio, transcript, speaker_id):
        if not transcript.strip():
            raise ValueError('Enter the exact words spoken in this reference.')
        return ({'audio': audio, 'text': transcript, 'speaker_id': speaker_id},)


class Drama3Voices(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Drama3Voices', display_name='Drama 3 - Voice Bank', category='audio/Drama 3',
            inputs=[io.Autogrow.Input('voices', template=io.Autogrow.TemplatePrefix(VOICE.Input('voice'), prefix='voice_', min=0, max=100))],
            outputs=[VOICES.Output(display_name='voices')])

    @classmethod
    def execute(cls, voices: io.Autogrow.Type = None):
        return io.NodeOutput(list((voices or {}).values()))


class Drama3TakeFromAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': ('AUDIO',), 'transcript': ('STRING', {'default': '', 'multiline': True})}}
    RETURN_TYPES = ('DRAMA3_TAKE',)
    FUNCTION = 'create'
    CATEGORY = 'audio/Drama 3'

    def create(self, audio, transcript):
        if not transcript.strip():
            raise ValueError('Enter the exact transcript of the previous take.')
        return ({'audio': audio, 'text': transcript},)


class Drama3SaveTake:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'take': ('DRAMA3_TAKE',), 'filename_prefix': ('STRING', {'default': 'Drama3/take'})}}
    RETURN_TYPES = ()
    FUNCTION = 'save'
    CATEGORY = 'audio/Drama 3'
    OUTPUT_NODE = True

    def save(self, take, filename_prefix):
        if 'audio' in take:
            raise ValueError('Save the generated take output; imported audio has not been encoded yet.')
        root = Path(folder_paths.get_output_directory()).resolve()
        target = (root / filename_prefix).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Save takes inside the ComfyUI output folder.')
        target.parent.mkdir(parents=True, exist_ok=True)
        path = target.with_name(target.name + '_' + uuid.uuid4().hex[:8]).with_suffix('.npz')
        save_take(path, take)
        return {'ui': {'text': [f'Saved reusable take: {path.relative_to(root)}']}}


class Drama3LoadTake:
    @classmethod
    def INPUT_TYPES(cls):
        root = Path(folder_paths.get_input_directory())
        files = sorted(str(path.relative_to(root)).replace('\\', '/') for path in root.rglob('*.npz'))
        return {'required': {'take_file': (files,)}}
    RETURN_TYPES = ('DRAMA3_TAKE',)
    FUNCTION = 'load'
    CATEGORY = 'audio/Drama 3'
    DESCRIPTION = 'Copy a saved Drama 3 .npz take into ComfyUI/input, then select it here.'

    @classmethod
    def IS_CHANGED(cls, take_file):
        path = cls.resolve(take_file)
        return (path.stat().st_mtime_ns, path.stat().st_size)

    @staticmethod
    def resolve(name):
        root = Path(folder_paths.get_input_directory()).resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != '.npz':
            raise ValueError('Select a .npz take inside ComfyUI/input.')
        return path

    def load(self, take_file):
        return (load_take(self.resolve(take_file)),)


Drama3Voices.GET_SCHEMA()
NODE_CLASS_MAPPINGS = {'Drama3FP8Speech': Drama3FP8Speech, 'Drama3Voice': Drama3Voice,
    'Drama3Voices': Drama3Voices, 'Drama3TakeFromAudio': Drama3TakeFromAudio,
    'Drama3SaveTake': Drama3SaveTake, 'Drama3LoadTake': Drama3LoadTake}
NODE_DISPLAY_NAME_MAPPINGS = {'Drama3FP8Speech': 'Drama 3 - FP8 Speech (RAM Offload)',
    'Drama3Voice': 'Drama 3 - Voice Reference', 'Drama3TakeFromAudio': 'Drama 3 - Previous Audio Take',
    'Drama3SaveTake': 'Drama 3 - Save Take', 'Drama3LoadTake': 'Drama 3 - Load Take'}
WEB_DIRECTORY = './web'
