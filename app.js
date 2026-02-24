// ================= CONFIG =================
const SIZE = 16;
const LR = 0.005;

let model;
let optimizer;
let xInput;
let auto = false;
let step = 0;

// =============== LOG ======================
function log(msg) {
  const el = document.getElementById("log");
  const div = document.createElement("div");
  div.textContent = "> " + msg;
  el.prepend(div);
}

// =============== DATA =====================
function createInput() {
  return tf.randomUniform([1, SIZE, SIZE, 1]);
}

// =============== MODEL ====================
function createModel(arch) {
  const m = tf.sequential();
  m.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));

  if (arch === "compression") {
    m.add(tf.layers.dense({ units: 64, activation: "relu" }));
  } else if (arch === "transformation") {
    m.add(tf.layers.dense({ units: 256, activation: "relu" }));
    m.add(tf.layers.dense({ units: 256, activation: "relu" }));
  } else if (arch === "expansion") {
    m.add(tf.layers.dense({ units: 512, activation: "relu" }));
    m.add(tf.layers.dense({ units: 512, activation: "relu" }));
  }

  m.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  m.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));

  return m;
}

// =============== LOSSES ===================
// Sorted MSE (LECTURE CORE)
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const t = yTrue.flatten();
    const p = yPred.flatten();

    const tSorted = tf.sort(t);
    const pSorted = tf.sort(p);

    return tf.losses.meanSquaredError(tSorted, pSorted);
  });
}

// Smoothness (Total Variation)
function smoothness(y) {
  return tf.tidy(() => {
    const dx = y.slice([0, 0, 0, 0], [-1, -1, SIZE - 1, -1])
      .sub(y.slice([0, 0, 1, 0], [-1, -1, SIZE - 1, -1]));

    const dy = y.slice([0, 0, 0, 0], [-1, SIZE - 1, -1, -1])
      .sub(y.slice([0, 1, 0, 0], [-1, SIZE - 1, -1, -1]));

    return tf.mean(dx.square()).add(tf.mean(dy.square()));
  });
}

// Direction: dark left → bright right
function directionLoss(y) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, SIZE)
      .reshape([1, 1, SIZE, 1]);
    return tf.mean(y.mul(mask)).mul(-1);
  });
}

function totalLoss(x, y) {
  const l1 = sortedMSE(x, y);     // CONSERVE COLORS (lecture!)
  const l2 = smoothness(y);       // SMOOTH GRADIENT
  const l3 = directionLoss(y);    // LEFT→RIGHT

  return tf.addN([
    l1.mul(5.0),
    l2.mul(2.0),
    l3.mul(1.0)
  ]);
}

// =============== TRAIN ====================
function trainStep() {
  step++;

  const lossTensor = optimizer.minimize(() => {
    const yPred = model.predict(xInput);
    return totalLoss(xInput, yPred);
  }, true);

  const loss = lossTensor.dataSync()[0];
  lossTensor.dispose();

  render();
  log(`Step ${step} | Loss: ${loss.toFixed(4)}`);
}

// =============== RENDER ===================
async function render() {
  const out = model.predict(xInput);

  await tf.browser.toPixels(
    xInput.squeeze(),
    document.getElementById("inputCanvas")
  );

  await tf.browser.toPixels(
    out.squeeze(),
    document.getElementById("outputCanvas")
  );

  out.dispose();
}

// =============== RESET ====================
function reset() {
  const arch = document.querySelector(
    'input[name="arch"]:checked'
  ).value;

  if (model) model.dispose();

  model = createModel(arch);
  optimizer = tf.train.adam(LR);
  step = 0;

  log(`Model reset. Architecture: ${arch}`);
  render();
}

// =============== AUTO TRAIN ===============
function toggleAuto() {
  auto = !auto;
  const btn = document.getElementById("autoBtn");
  btn.textContent = auto ? "Auto Train (Stop)" : "Auto Train (Start)";

  if (auto) loop();
}

function loop() {
  if (!auto) return;
  trainStep();
  setTimeout(loop, 50);
}

// =============== INIT =====================
async function init() {
  xInput = createInput();

  document.getElementById("trainBtn").onclick = trainStep;
  document.getElementById("autoBtn").onclick = toggleAuto;
  document.getElementById("resetBtn").onclick = reset;

  document.querySelectorAll('input[name="arch"]').forEach(r => {
    r.onchange = reset;
  });

  reset();
  log("Stable version initialized. Buttons fixed.");
}

init();
