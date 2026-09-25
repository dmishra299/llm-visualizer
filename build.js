import { build } from "esbuild";
import { readFileSync, writeFileSync } from "fs";

const bundle = await build({
  entryPoints: ["js/app.js"],
  bundle: true,
  format: "iife",
  write: false,
});

const js  = bundle.outputFiles[0].text;
const css = readFileSync("styles.css", "utf8");
const template = readFileSync("index.html", "utf8");

const standalone = template
  .replace('<link rel="stylesheet" href="styles.css" />', `<style>\n${css}</style>`)
  .replace('<script type="module" src="js/app.js"></script>', `<script>\n${js}</script>`);

writeFileSync("standalone.html", standalone);
console.log(`standalone.html built (${(Buffer.byteLength(standalone) / 1024).toFixed(1)} KB)`);
