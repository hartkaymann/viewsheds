import { Controller } from "../Controller";

export class WorkspaceManager {

    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    async init() {
        document.getElementById("info-button")?.addEventListener("click", this.activateWorkspace.bind(this, "info"));
        document.getElementById("load-button")?.addEventListener("click", this.activateWorkspace.bind(this, "load"));
        document.getElementById("process-button")?.addEventListener("click", this.activateWorkspace.bind(this, "process"));
    }

    async activateWorkspace(workspaceId?: string) {
        const workspaces = document.querySelectorAll(".workspace");
        workspaces.forEach((workspace) => {
            if (workspace.id !== `workspace-${workspaceId}`) {
                workspace.classList.remove("active");
            }
        });        
        document.getElementById(`workspace-${workspaceId}`)?.classList.add("active");

        const toolbarButtons = document.querySelectorAll(".toolbar-button");
        toolbarButtons.forEach((button) => {
            if (button.id !== `${workspaceId}-button`) {
                button.classList.remove("active");
            }
        });        
        document.getElementById(`${workspaceId}-button`)?.classList.add("active");
    }
}