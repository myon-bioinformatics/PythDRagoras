"use strict";
/** Thin Node entry (Node 26+): runs score_sensitivity.py through the shared vendor/py_launch.cjs. */
const path = require("node:path");
const { main } = require("./vendor/py_launch.cjs");

const PY_SCRIPT = path.join(__dirname, "score_sensitivity.py");
const PY_LAUNCH = { envVar: "SCORE_SENSITIVITY_PYTHON", noSite: true };

if (require.main === module) {
  main(PY_SCRIPT, PY_LAUNCH);
}

module.exports = { PY_SCRIPT, PY_LAUNCH };
