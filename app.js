/**
 * FINAL STABLE VERSION
 * - Работает на GitHub Pages
 * - Кнопки не ломаются
 * - Нет PointerEvent бага
 * - Нет "Cannot find connection" ошибки
 * - Чистый Sorted MSE (как в лекции)
 * - Строит градиент стабильно
 */

// ================= CONFIG =================
const SIZE = 16;
const LEARNING_RATE = 0.005;
const AUTO_DELAY = 50;

let state = {
  step: 0,
  isAuto: false,
  model: null,
  optimizer: null,
  xInput: null
};

// ================= LOG =================
function log(msg) {
  const logEl = document.getElementById("log");
  const line = document.createElement("div");
  line.textContent = "> " + msg;
  logEl.prepend(line);
}

// ================= DATA =================
function createFixedNoise() {
  return tf.randomUniform([1, SIZE, SIZE, 1]);
}

// ================= MODEL =================
function createStudentModel(arch) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));

  if (arch === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  } else if (arch === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
  } else if (arch === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
  }

  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));

  return model;
}

// ================= LOSSES (LECTURE-ALIGNED) =================

// 🔥 Sorted MSE = Quantile / 1D Wasserstein (как в лекции)
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const t = yTrue.reshape([SIZE * SIZE]);
    const p = yPred.reshape([SIZE * SIZE]);

    const tSorted = tf.sort(t);
    const pSorted = tf.sort(p);

    return tf.losses.meanSquaredError(tSorted, pSorted);
  });
}

// Total Variation (гладкость)
function smoothnessLoss(y) {
  return tf.tidy(() => {
    const dx = y.slice([0, 0, 0, 0], [-1, -1, SIZE - 1, -1])
      .sub(y.slice([0, 0, 1, 0], [-1, -1, SIZE - 1, -1]));

    const dy = y.slice([0, 0, 0, 0], [-1, SIZE - 1, -1, -1])
      .sub(y.slice([0, 1, 0, 0], [-1, SIZE - 1, -1, -1]));

    return tf.mean(dx.square()).add(tf.mean(dy.square()));
  });
}

// Направление: слева тёмно → справа светло
function directionLoss(y) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, SIZE)
      .reshape([1, 1, SIZE, 1]);

    return tf.mean(y.mul(mask)).mul(-1);
  });
}

// Финальный loss (ПРЯМО ПО СЛАЙДУ)
function studentLoss(xInput, yPred) {
  return tf.tidy(() => {
    const lSorted = sortedMSE(xInput, yPred); // ключ лекции
    const lSmooth = smoothnessLoss(yPred);
    const lDir = directionLoss(yPred);

    return tf.addN([
      lSorted.mul(5.0),   // сохраняем "инвентарь цветов"
      lSmooth.mul(2.0),   // делаем градиент гладким
      lDir.mul(1.0)       // задаём направление
    ]);
  });
}

// ================= TRAIN =================
function trainStep() {
  state.step++;

  const lossTensor = state.optimizer.minimize(() => {
    const yPred = state.model.predict(state.xInput);
    return studentLoss(state.xInput, yPred);
  }, true);

  const lossValue = lossTensor.dataSync()[0];
  lossTensor.dispose();

  render();
  log(`Step ${state.step} | Loss: ${lossValue.toFixed(4)}`);
}

// ================= AUTO TRAIN =================
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

// ================= RENDER =================
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

// ================= RESET =================
function getSelectedArch() {
  const radio = document.querySelector('input[name="arch"]:checked');
  return radio ? radio.value : "compression";
}

function resetModels() {
  if (state.model) state.model.dispose();

  const arch = getSelectedArch();

  state.model = createStudentModel(arch);
  state.optimizer = tf.train.adam(LEARNING_RATE);
  state.step = 0;

  log(`Models reset. Arch: ${arch}`);
  render();
}

// ================= INIT =================
function init() {
  state.xInput = createFixedNoise();

  // КНОПКИ (без багов и PointerEvent)
  document.getElementById("trainBtn").addEventListener("click", trainStep);
  document.getElementById("autoBtn").addEventListener("click", toggleAuto);
  document.getElementById("resetBtn").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.addEventListener("change", resetModels);
  });

  resetModels();
  log("FINAL stable version initialized (GitHub Pages safe).");
}

init();
