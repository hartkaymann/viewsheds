import { LasHeaderSummary } from "../SceneLoader";

/**
 * Owns the import UI inside the "Load" workspace: file picker, LAZ header
 * readout, a confirm button and a stack of per-stage progress bars (one bar
 * per loading step, created the first time that step reports progress).
 */
export class ImportControls {
    private fileInput: HTMLInputElement;
    private info: HTMLElement;
    private confirmButton: HTMLButtonElement;
    private progressWrapper: HTMLElement;

    /** One progress-bar element per loading stage, in arrival order. */
    private stageBars: Map<string, HTMLElement> = new Map();

    private fileSelectedCb: ((file: File) => void) | null = null;
    private confirmCb: (() => void) | null = null;

    constructor() {
        this.fileInput = document.getElementById("file-input") as HTMLInputElement;
        this.info = document.getElementById("file-info") as HTMLElement;
        this.confirmButton = document.getElementById("confirm-load") as HTMLButtonElement;
        this.progressWrapper = document.getElementById("import-progress") as HTMLElement;

        this.fileInput.addEventListener("change", () => {
            const file = this.fileInput.files?.[0];
            if (file) {
                this.reset();
                this.fileSelectedCb?.(file);
            }
        });

        this.confirmButton.addEventListener("click", () => {
            this.confirmButton.disabled = true;
            this.confirmCb?.();
        });

        this.reset();
    }

    onFileSelected(cb: (file: File) => void) {
        this.fileSelectedCb = cb;
    }

    onConfirm(cb: () => void) {
        this.confirmCb = cb;
    }

    showHeader(summary: LasHeaderSummary) {
        const rows: [string, string][] = [
            ["File", summary.fileName],
            ["Version", `${summary.versionMajor}.${summary.versionMinor}`],
            ["Points", summary.pointCount.toLocaleString()],
            ["Format", `${summary.formatId} (${summary.isCompressed ? "compressed" : "uncompressed"})`],
            ["Record length", `${summary.pointDataRecordLength} bytes`],
            ["Size", ImportControls.formatBytes(summary.fileSize)],
        ];

        this.info.innerHTML = rows
            .map(([k, v]) => `<div class="file-info-row"><span>${k}</span><span>${v}</span></div>`)
            .join("");

        this.confirmButton.disabled = false;
    }

    setProgress(stage: string, value: number) {
        this.progressWrapper.style.display = "flex";
        const bar = this.getOrCreateStageBar(stage);
        bar.style.width = `${Math.round(value * 100)}%`;
        bar.classList.toggle("complete", value >= 1);
    }

    /** Mark every stage seen so far as finished. */
    complete() {
        for (const bar of this.stageBars.values()) {
            bar.style.width = "100%";
            bar.classList.add("complete");
        }
    }

    reset() {
        this.info.innerHTML = "";
        this.confirmButton.disabled = true;
        this.progressWrapper.style.display = "none";
        this.progressWrapper.innerHTML = "";
        this.stageBars.clear();
    }

    private getOrCreateStageBar(stage: string): HTMLElement {
        const existing = this.stageBars.get(stage);
        if (existing) return existing;

        const row = document.createElement("div");
        row.className = "progress-stage";

        const label = document.createElement("div");
        label.className = "progress-label";
        label.textContent = stage;

        const track = document.createElement("div");
        track.className = "progress";

        const bar = document.createElement("div");
        bar.className = "progress-bar";

        track.appendChild(bar);
        row.appendChild(label);
        row.appendChild(track);
        this.progressWrapper.appendChild(row);

        this.stageBars.set(stage, bar);
        return bar;
    }

    private static formatBytes(bytes: number): string {
        if (bytes < 1024) return `${bytes} B`;
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(1)} KB`;
        const mb = kb / 1024;
        if (mb < 1024) return `${mb.toFixed(1)} MB`;
        return `${(mb / 1024).toFixed(2)} GB`;
    }
}
