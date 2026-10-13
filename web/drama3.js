import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

const previews = new WeakMap();
app.registerExtension({
    name: 'Drama3.ProgressiveAudio',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'Drama3FP8Speech') return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previous?.apply(this, arguments);
            const panel = document.createElement('div');
            panel.style.cssText = 'display:flex;flex-direction:column;gap:5px;padding:5px;color:var(--input-text);font:12px sans-serif';
            const status = document.createElement('div');
            status.textContent = 'Progressive preview: enable streaming to hear parts while generating.';
            const button = document.createElement('button');
            button.textContent = 'Enable live listening';
            const player = document.createElement('audio');
            player.controls = true;
            player.style.cssText = 'width:100%;height:32px';
            const state = { status, player, button, enabled: false, queue: [], session: null };
            const playNext = () => {
                if (!state.enabled || !state.queue.length) return;
                player.src = state.queue.shift();
                player.play().catch(() => { status.textContent = 'Click the audio play button to allow playback.'; });
            };
            player.addEventListener('ended', playNext);
            button.onclick = () => {
                state.enabled = !state.enabled;
                button.textContent = state.enabled ? 'Pause live listening' : 'Enable live listening';
                if (state.enabled) {
                    if (player.src && !player.ended) player.play().catch(() => {});
                    else playNext();
                } else player.pause();
            };
            state.playNext = playNext;
            panel.append(status, button, player);
            this.addDOMWidget('drama3_preview', 'custom', panel, { serialize: false, getMinHeight: () => 100, getMaxHeight: () => 120 });
            previews.set(this, state);
        };
    },
});

api.addEventListener('drama3.preview', ({ detail }) => {
    const node = app.graph?.getNodeById(detail.node_id);
    const state = node && previews.get(node);
    if (!state) return;
    if (state.session !== detail.session) {
        state.session = detail.session;
        state.queue = [];
        state.player.pause();
        state.player.removeAttribute('src');
    }
    if (detail.stage) state.status.textContent = detail.stage;
    if (!detail.audio) return;
    const params = new URLSearchParams({ filename: detail.audio, type: 'temp', subfolder: detail.subfolder || '' });
    const url = api.apiURL('/view?' + params);
    state.queue.push(url);
    if (state.enabled && (state.player.ended || !state.player.src)) state.playNext();
    else if (!state.enabled && !state.player.src) state.player.src = state.queue.shift();
});
