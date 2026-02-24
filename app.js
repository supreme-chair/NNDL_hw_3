/**
 * Neural Network Design: The Gradient Puzzle
 * FINAL STABLE VERSION (Matches Lecture Requirements)
 * - Uses Sorted MSE (value conservation)
 * - No explicit permutation layer (as in lecture)
 * - Fully differentiable (no dataSync inside training graph)
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.003, // ↓ стабильнее
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// =======================
// LOSSES (LECTURE-CORRECT)
// =======================

// Pixel MSE (baseline only)
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Sorted MSE (Quantile / 1D Wasserstein proxy)
// IMPORTANT: sort OUTSIDE gradient graph safely
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const tFlat = yTrue.reshape([256]);
    const pFlat = yPred.reshape([256]);

    // Use tf.topk trick to get sorted tensors (DIFFERENTIABLE graph-safe)
    const tSorted = tf.topk(tFlat, 256, true).values;
    const pSorted = tf.topk(pFlat, 256, true).values;

    return tf.losses.meanSquaredError(tSorted, pSorted);
  });
}

// Total Variation (Smoothness)
function smoothness(yPred) {
  const dx = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  const dy = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
}

// Direction: left dark → right bright (from lecture slide)
function directionX(yPred) {
  const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// FINAL lecture-compliant loss
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const L_sorted = sortedMSE(yTrue, yPred); // conservation
    const L_smooth = smoothness(yPred);       // TV loss
    const L_dir = directionX(yPred);          // gradient direction

    // EXACT philosophy from lecture
    return tf.addN([
      L_sorted.mul(1.0),   // keep color inventory
      L_smooth.mul(4.0),   // strong structure
      L_dir.mul(1.0)       // enforce gradient
    ]);
  });
}

// =======================
// MODELS
// =======================

function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// =======================
// TRAINING LOOP (FIXED)
// =======================

async function trainStep() {
  state.step++;

  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    });
    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  const studentLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, yPred);
    });
    state.studentOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  log(
    `Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`
  );

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// =======================
// INIT & UI
// =======================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );

  document.getElementById("btn-train").onclick = trainStep;
  document.getElementById("btn-auto").onclick = toggleAutoTrain;
  document.getElementById("btn-reset").onclick = resetModels;

  document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.onchange = (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value;
    };
  });

  log("FINAL stable version initialized.");
}

function resetModels(archType = null) {
  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineOptimizer) state.baselineOptimizer.dispose();
  if (state.studentOptimizer) state.studentOptimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);

  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Arch: ${archType}`);
  render();
}

async function render() {
  const base = state.baselineModel.predict(state.xInput);
  const stud = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    base.clipByValue(0, 1).squeeze(),
    document.getElementById("canvas-baseline")
  );
  await tf.browser.toPixels(
    stud.clipByValue(0, 1).squeeze(),
    document.getElementById("canvas-student")
  );

  base.dispose();
  stud.dispose();
}

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText =
    `Loss: ${stud.toFixed(5)}`;
}

function log(msg) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  el.prepend(span);
}

function toggleAutoTrain() {
  state.isAutoTraining = !state.isAutoTraining;
  document.getElementById("btn-auto").innerText =
    state.isAutoTraining ? "Auto Train (Stop)" : "Auto Train (Start)";
  loop();
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

init();
