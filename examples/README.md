# Generated speech examples

**[▶ Play all examples in the audio gallery](https://nebuluss.github.io/Drama-3/)** — multilingual speech, the corrected two-reference dialogue, and delivery cues.

All audio here is **AI-generated speech**. These are demonstrations of the local pipeline, not original statements by a speaker or a measured cloning-quality benchmark. WAV files are 44.1 kHz mono, 16-bit PCM.

## French, Spanish and Italian from a French reference

An 8.35-second French recording supplied locally by the repository author conditioned these three examples. The original recording and its reference VQ codes are not distributed. The example output is newly generated speech.

| Language | Audio | New text |
|---|---|---|
| Français — 7.52 s | [Listen / download](generated-fr-french-reference.wav) | Ce matin, j'ai décidé de ralentir un peu. Un bon café et une conversation tranquille, c'est déjà une belle journée. |
| Español — 7.29 s | [Listen / download](generated-es-french-reference.wav) | Esta mañana he decidido ir un poco más despacio. Un buen café y una conversación tranquila pueden cambiar todo el día. |
| Italiano — 8.08 s | [Listen / download](generated-it-french-reference.wav) | Questa mattina ho deciso di rallentare un po'. Un buon caffè e una conversazione tranquilla possono cambiare tutta la giornata. |

Settings: seed 46, original FP8 checkpoint, `direction=warm, conversational`, `chunk_mode=clauses`, `max_frames=512`, one reference. All eight clauses ended with EOS. One model load produced all three languages: 22.89 seconds of output in 161.74 seconds total, including loading, reference encoding and decoding. Peak PyTorch GPU allocation was 25.43 GiB.

[Full multilingual settings and measurements](multilingual-settings.json). Language choice follows the text; pronunciation and accent are model outputs and have not been scored by native speakers. The public cloning workflows start with a shipped synthetic reference, so loading those defaults produces a different voice from these locally conditioned examples.

## Two cloned references in a French dialogue

[Listen to the two-voice French conversation](two-distinct-clones-fr.wav).

Speaker 0 is conditioned on the author's local Raphaël Quenard reference; speaker 1 is conditioned on `benjamintranie_04.mp3`, a separate Benjamin Tranié recording selected by the author. Both references are recorded French speech. The four turns request curious, amused, warm and excited delivery. The 11.94-second conversation was regenerated as one native dialogue with seed 48; the earlier synthetic-reference version has been replaced. Original reference recordings are not distributed.

[Settings and measurements](two-distinct-settings.json). To reproduce the routing, load the multi-voice workflow and provide your own two references and exact transcripts.

## Delivery cues and seed variations

The following four clips are extracted from one complete multi-reference take. The file labels name the **requested** cue; their perceptual strength can vary.

| Requested cue | Audio | Target text |
|---|---|---|
| Warm, cheerful | [Listen](mood-warm.wav) | Hello, welcome back. |
| Angry | [Listen](mood-angry.wav) | You are late again! |
| Sad, disappointed | [Listen](mood-sad.wav) | I am sorry about that. |
| Whisper | [Listen](mood-whisper.wav) | It is fine. Come inside. |

Complete takes: [seed 42](dialogue-seed42.wav), [seed 43](dialogue-seed43.wav). These use two related synthetic reference clips and four separately generated turns. The two variations share one model load. The whole batch took 129.02 seconds; peak PyTorch allocation was 26.45 GiB with progressive decoding enabled.

## Starter assets

- [Drama3_demo_reference.wav](Drama3_demo_reference.wav) — synthetic speech: “Hello, this is a local test.”
- [Drama3_demo_reference_2.wav](Drama3_demo_reference_2.wav) — a synthetic reference-conditioned continuation: “This voice comes from the reference clip.”
- [Drama3_demo_take.npz](Drama3_demo_take.npz) — a reusable generated take with the two synthetic references, for the Load Saved Take workflow.
- [French comma-separated clauses](french-clauses-synthetic-reference.wav) — “Bonjour, je teste le clonage de voix, et cette phrase doit aller jusqu au bout.” This test reuses saved context and overrides one reference.

Copy the three `Drama3_demo_*` files into `ComfyUI/input` for the starter workflows. The second source clip is related to the first; use two independent references when evaluating speaker separation.

## Validation notes

[Machine measurements](validation.json) include the initial dialogue batch, saved-take/clauses test and a longer streaming stress test. The latter delivered blocks at 64, 128, 192 and 256 frames and reached its 256-frame cap (`length`); that truncated stress output is not offered as a complete speech demo.

The tests used a 32 GB RTX 5090, 128 GB system RAM, PyTorch 2.8.0 / CUDA 12.9 and Triton Windows 3.4.0.post21. Reported GPU numbers are PyTorch allocation peaks, not the total desktop-wide memory reading. No model conversion, fine-tuning or external speech-generation service was used.
