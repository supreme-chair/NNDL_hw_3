/**
 * FINAL GITHUB PAGES SAFE VERSION
 * - No tf.sort (browser compatible)
 * - Stable training
 * - Buttons always work
 * - Matches lecture: Sorted MSE + Smooth + Direction
 */

const SIZE = 16;
const LR = 0.005;
const AUTO_DELAY = 50;

let state = {
  step: 0,
  isAuto: false,
  model: null,
  optimizer: null,
  xInput: null
};

// ===== LOG =====
function log(msg) {
  const el = document.getElementById("log");
  const line = document.createElement("div");
  line.textContent = "> " + msg;
  el.prepend(line);
}

// ===== DATA =====
function createNoise() {
  return tf.randomUniform([1, SIZE, SIZE, 1]);
}

// ===== MODEL =====
function createModel(arch) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));

  if (arch === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  } 
  else if (arch === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
  } 
  else if (arch === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
  }

  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));

  return model;
}

// ===== SORTED MSE (LECTURE-CORRECT, TFJS SAFE) =====
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const tFlat = yTrue.reshape([SIZE * SIZE]);
    const pFlat = yPred.reshape([SIZE * SIZE]);

    // 🔥 ВАЖНО: JS sort вместо tf.sort (иначе краш в браузере)
    const tArr = Array.from(tFlat.dataSync()).sort((a, b) => a - b);
    const pArr = Array.from(pFlat.dataSync()).sort((a, b) => a - b);

    const tSorted = tf.tensor1d(tArr);
    const pSorted = tf.tensor1d(pArr);

    const loss = tf.losses.meanSquaredError(tSorted, pSorted);

    tSorted.dispose();
    pSorted.dispose();

    return loss;
  });
}

// ===== SMOOTHNESS (TV Loss) =====
function smoothnessLoss(y) {
  return tf.tidy(() => {
    const dx = y.slice([0, 0, 0, 0], [-1, -1, SIZE - 1, -1])
      .sub(y.slice([0, 0, 1, 0], [-1, -1, SIZE - 1, -1]));

    const dy = y.slice([0, 0, 0, 0], [-1, SIZE - 1, -1, -1])
      .sub(y.slice([0, 1, 0, 0], [-1, SIZE - 1, -1, -1]));

    return tf.mean(dx.square()).add(tf.mean(dy.square()));
  });
}

// ===== DIRECTION LOSS =====
function directionLoss(y) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, SIZE)
      .reshape([1, 1, SIZE, 1]);

    return tf.mean(y.mul(mask)).mul(-1);
  });
}

// ===== FINAL LOSS (как на слайде) =====
function studentLoss(xInput, yPred) {
  return tf.tidy(() => {
    const lSorted = sortedMSE(xInput, yPred);
    const lSmooth = smoothnessLoss(yPred);
    const lDir = directionLoss(yPred);

    return tf.addN([
      lSorted.mul(2.0),   // ключ: сохранить "инвентарь цветов"
      lSmooth.mul(4.5),   // сгладить в градиент
      lDir.mul(1.0)       // направление слева → направо
    ]);
  });
}

// ===== TRAIN =====
function trainStep() {
  state.step++;

  const lossTensor = state.optimizer.minimize(() => {
    const yPred = state.model.predict(state.xInput);
    return studentLoss(state.xInput, yPred);
  }, true);

  const loss = lossTensor.dataSync()[0];
  lossTensor.dispose();

  render();
  log(`Step ${state.step} | Loss: ${loss.toFixed(4)}`);
}

// ===== AUTO TRAIN =====
function autoLoop() {
  if (!state.isAuto) return;
  trainStep();
  setTimeout(autoLoop, AUTO_DELAY);
}

function toggleAuto() {
  state.isAuto = !state.isAuto;
  const btn = document.getElementById("autoBtn");
  btn.textContent = state.isAuto ? "Auto Train (Stop)" : "Auto Train (Start)";
  if (state.isAuto) autoLoop();
}

// ===== RENDER =====
async function render() {
  const yPred = state.model.predict(state.xInput);

  await tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );

  await tf.browser.toPixels(
    yPred.squeeze(),
    document.getElementById("canvas-student")
  );

  yPred.dispose();
}

// ===== RESET =====
function getArch() {
  const radio = document.querySelector('input[name="arch"]:checked');
  return radio ? radio.value : "compression";
}

function resetModel() {
  if (state.model) state.model.dispose();

  const arch = getArch();
  state.model = createModel(arch);
  state.optimizer = tf.train.adam(LR);
  state.step = 0;

  log(`Model reset | Arch: ${arch}`);
  render();
}

// ===== INIT =====
function init() {
  state.xInput = createNoise();

  document.getElementById("trainBtn").addEventListener("click", trainStep);
  document.getElementById("autoBtn").addEventListener("click", toggleAuto);
  document.getElementById("resetBtn").addEventListener("click", resetModel);

  document.querySelectorAll('input[name="arch"]').forEach(r => {
    r.addEventListener("change", resetModel);
  });

  resetModel();
  log("GitHub Pages stable build initialized.");
}

init();
