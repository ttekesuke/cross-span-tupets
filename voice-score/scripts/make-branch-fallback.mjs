import { copyFile, mkdir, readdir, readFile, writeFile } from "node:fs/promises";

const distDir = "dist-pages";
const assetDir = `${distDir}/assets`;
const assetNames = await readdir(assetDir);
const appName = assetNames.find((name) => /^index-.*\.js$/.test(name));
const osmdName = assetNames.find((name) => /^opensheetmusicdisplay\.min-.*\.js$/.test(name));
const cssName = assetNames.find((name) => /^index-.*\.css$/.test(name));
const analysisWorkerName = assetNames.find((name) => /^analysis\.worker-.*\.js$/.test(name));
const transcribeWorkerName = assetNames.find((name) => /^transcribe\.worker-.*\.js$/.test(name));
if (!appName || !osmdName || !cssName || !analysisWorkerName || !transcribeWorkerName) {
  throw new Error("Viteの静的アセットを特定できませんでした");
}

const appPath = `${assetDir}/${appName}`;
const app = await readFile(appPath, "utf8");
// Keep dist-pages untouched: the Actions deployment includes the bundled
// OSMD chunk. Only the branch-publishing fallback uses the CDN wrapper.
await mkdir("pages-static-assets", { recursive: true });
await writeFile(`pages-static-assets/${appName}`, app.replaceAll(`./${osmdName}`, "./osmd-wrapper.js"));
await writeFile("pages-static-assets/osmd-wrapper.js", `
const api = globalThis.opensheetmusicdisplay;
export const OpenSheetMusicDisplay = api.OpenSheetMusicDisplay;
export default api;
`);

const index = await readFile(`${distDir}/index.html`, "utf8");
const withExternalOsmd = index.replace(
  "</head>",
  "    <script src=\"https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@2.1.2/build/opensheetmusicdisplay.min.js\"></script>\n  </head>",
);
await copyFile("source-index.html", "index.html");
await writeFile("index.html", withExternalOsmd);
await copyFile(`${assetDir}/${cssName}`, `pages-static-assets/${cssName}`);
await copyFile(`${assetDir}/${analysisWorkerName}`, `pages-static-assets/${analysisWorkerName}`);
await copyFile(`${assetDir}/${transcribeWorkerName}`, `pages-static-assets/${transcribeWorkerName}`);

