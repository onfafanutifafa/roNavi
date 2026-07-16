import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "server/proxy": "src/server/proxy.ts",
    "cli/cli": "src/cli/cli.ts",
  },
  format: ["esm"],
  target: "node18",
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: false,
  // Preserve the shebang on the CLI entry so `ronavi` is executable.
  banner: {},
});
