import { copyFile } from "node:fs/promises";

// Keep the repository's index.html usable when GitHub Pages is configured to
// publish the main branch, while restoring the Vite source entry for builds.
await copyFile("source-index.html", "index.html");

