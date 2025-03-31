interface RayDataTableConfig {
    containerId: string;
    rayCount: number;
    blockSize: number;
    indexArray: Uint32Array;
    distanceArray: Float32Array;
    raysPerPage?: number;
}

export class RayDataTable {
    private container: HTMLElement;
    private rayCount: number;
    private blockSize: number;
    private indexArray: Uint32Array;
    private distanceArray: Float32Array;
    private raysPerPage: number;
    private currentPage: number;

    constructor(config: RayDataTableConfig) {
        this.container = document.getElementById(config.containerId)!;
        this.rayCount = config.rayCount;
        this.blockSize = config.blockSize;
        this.indexArray = config.indexArray;
        this.distanceArray = config.distanceArray;
        this.raysPerPage = config.raysPerPage || 10;
        this.currentPage = 1;
    }

    displayRayData() {
        if (!this.container) return;

        this.container.innerHTML = "";
        this.container.className = "debug-table-container";

        const scrollWrapper = document.createElement("div");
        scrollWrapper.className = "debug-table-scroll-wrapper";

        const contentTable = document.createElement("table");
        contentTable.className = "debug-table";

        // Create header row directly in the main table
        const header = document.createElement("tr");
        ["Ray", "Node Index", "Distance"].forEach(text => {
            const th = document.createElement("th");
            th.textContent = text;
            th.className = "debug-table-header";
            header.appendChild(th);
        });
        contentTable.appendChild(header);

        const startRay = (this.currentPage - 1) * this.raysPerPage;
        const endRay = Math.min(startRay + this.raysPerPage, this.rayCount);

        for (let ray = startRay; ray < endRay; ray++) {
            const offset = ray * this.blockSize;
            const rows: HTMLTableRowElement[] = [];
            let i = 0;

            while (i < this.blockSize) {
                const nodeIndex = this.indexArray[offset + i];
                if (nodeIndex === 0) {
                    let count = 0;
                    while (i < this.blockSize && this.indexArray[offset + i] === 0) count++, i++;
                    const row = document.createElement("tr");
                    row.className = "debug-table-row";
                    row.insertCell().textContent = `-- x${count}`;
                    row.insertCell().textContent = "--";
                    rows.push(row);
                } else {
                    const row = document.createElement("tr");
                    row.className = "debug-table-row";
                    row.insertCell().textContent = nodeIndex.toString();
                    row.insertCell().textContent = this.distanceArray[offset + i].toFixed(3);
                    rows.push(row);
                    i++;
                }
            }

            if (rows.length > 0) {
                const rayCell = document.createElement("td");
                rayCell.textContent = ray.toString();
                rayCell.rowSpan = rows.length;
                rayCell.className = "debug-table-ray-index";
                rows[0].insertBefore(rayCell, rows[0].firstChild);
            }

            rows.forEach((row, index) => {
                const isLast = index === rows.length - 1;
                const isFirst = index === 0;
                if (isFirst) row.classList.add("debug-table-first-row");
                if (isLast) row.classList.add("debug-table-last-row");
                contentTable.appendChild(row);
            });
        }

        scrollWrapper.appendChild(contentTable);
        this.container.appendChild(scrollWrapper);

        this.createPaginationControls();
    }

    private createPaginationControls() {
        const paginationContainer = document.createElement("div");
        paginationContainer.className = "pagination-container";

        const totalPages = Math.ceil(this.rayCount / this.raysPerPage);

        const addPageButton = (label: string, page: number | null, disabled: boolean, isActive = false) => {
            const btn = document.createElement("button");
            btn.textContent = label;
            btn.className = "pagination-button";
            if (isActive) btn.classList.add("active");
            btn.disabled = disabled;
            if (page !== null && !disabled) {
                btn.addEventListener("click", () => {
                    this.currentPage = page;
                    this.displayRayData();
                });
            }
            paginationContainer.appendChild(btn);
        };

        addPageButton("<<", 1, this.currentPage === 1);
        addPageButton("<", this.currentPage - 1, this.currentPage === 1);

        const pageWindow = 2;
        for (let i = this.currentPage - pageWindow; i <= this.currentPage + pageWindow; i++) {
            if (i < 1 || i > totalPages) {
                addPageButton("", null, true);
            } else {
                addPageButton(i.toString(), i, false, i === this.currentPage);
            }
        }

        addPageButton(">", this.currentPage + 1, this.currentPage === totalPages);
        addPageButton(">>", totalPages, this.currentPage === totalPages);

        this.container.appendChild(paginationContainer);
    }
}
