/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Modify the Student Model architecture and loss function to transform
 * random noise input into a smooth, directional gradient output.
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  // Model definition shape (no batch dim) - used for layer creation
  inputShapeModel: [16, 16, 1],
  // Data tensor shape (includes batch dim) - used for input tensor creation
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.005,
  autoTrainSpeed: 50, // ms delay between steps (lower is faster)
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null, // The fixed noise input
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard MSE: Mean Squared Error
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// TODO: Helper - Smoothness (Total Variation)
// Penalize differences between adjacent pixels to encourage smoothness.
// Safe argsort for TFJS (browser compatible)
function argsort1D(tensor) {
  const arr = Array.from(tensor.dataSync())
    .map((v, i) => ({ v, i }))
    .sort((a, b) => a.v - b.v)
    .map(obj => obj.i);

  return tf.tensor1d(arr, 'int32');
}

// HARD permutation layer (NO new colors!)
function rearrangeByScores(xInput, scores) {
  return tf.tidy(() => {
    const xFlat = xInput.reshape([256]);   // ORIGINAL pixels
    const sFlat = scores.reshape([256]);   // predicted ordering scores

    // JS argsort (instead of tf.argsort)
    const indices = argsort1D(sFlat);

    // 🔥 CRITICAL: we gather ONLY original pixels
    const rearranged = tf.gather(xFlat, indices);

    return rearranged.reshape([1, 16, 16, 1]);
  });
}
function smoothness(yPred) {
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}
// TODO: Helper - Directionality (Gradient)
// Encourage pixels on the right to be brighter than pixels on the left.
function directionX(yPred) {
  // Create a weight mask that increases from left (-1) to right (+1)
  // For 16x16, we can just use linspace
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]); // [1, 1, 16, 1]

  // We want yPred to correlate with mask.
  // Maximize (yPred * mask) => Minimize -(yPred * mask)
  return tf.mean(yPred.mul(mask)).mul(-1);
}
// Helper: Safe sort for TensorFlow.js (browser compatible)
function softHistogramLoss(yTrue, yPred, bins = 32) {
  return tf.tidy(() => {
    const yT = yTrue.reshape([-1]);
    const yP = yPred.reshape([-1]);

    const min = 0.0;
    const max = 1.0;
    const binWidth = (max - min) / bins;

    let loss = tf.scalar(0);
    for (let i = 0; i < bins; i++) {
      const center = min + i * binWidth + binWidth / 2;

      const tDist = tf.exp(yT.sub(center).square().div(-0.01));
      const pDist = tf.exp(yP.sub(center).square().div(-0.01));

      const tHist = tf.mean(tDist);
      const pHist = tf.mean(pDist);

      loss = loss.add(tHist.sub(pHist).square());
    }
    return loss;
  });
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression (Undercomplete AE)
// 16x16 -> 64 -> 16x16
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Bottleneck
  model.add(tf.layers.dense({ units: 256, activation: "linear" })); // Output 0-1
  // Reshape back to [16, 16, 1] (batch dim is handled automatically)
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------------
// [TODO-A]: STUDENT ARCHITECTURE DESIGN
// Modify this function to implement 'transformation' and 'expansion'.
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // Undercomplete projection (как baseline, но обучается с другим loss)
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "linear" }));

  } else if (archType === "transformation") {
    // Projection 256 → 256 (чистая трансформация)
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "linear" }));

  } else if (archType === "expansion") {
    // Overcomplete projection (как в лекции)
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "linear" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}


// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  // Safety check: Ensure models are initialized
  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (MSE Only)
  // We use a simple fit here or gradient tape, let's use tape for consistency
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred); // Baseline always uses MSE
    }, state.baselineModel.getWeights());

    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        // модель генерирует output напрямую (дифференцируемо)
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());

      state.studentOptimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
    log(
      `Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`,
    );
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Visualize
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization logic
// ==========================================

function init() {
  // 1. Generate fixed noise (Batch size included: [1, 16, 16, 1])
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // 2. Initialize Models
  resetModels();

  // 3. Render Initial Input
  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  // 4. Bind Events
  document
    .getElementById("btn-train")
    .addEventListener("click", () => trainStep());
  document
    .getElementById("btn-auto")
    .addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Ready to train.");
}

function resetModels(archType = null) {
  // [Fix]: When called via event listener, archType is an Event object.
  // We must ensure it's either a string or null.
  if (typeof archType !== "string") {
    archType = null;
  }

  // Safety: Stop auto-training to prevent race conditions during reset
  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Dispose old resources to avoid memory leaks
  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  // Important: Dispose optimizer because it holds references to old model variables.
  if (state.baselineOptimizer) {
  state.baselineOptimizer.dispose();
  }
  if (state.studentOptimizer) {
  state.studentOptimizer.dispose();
  }

  // Create New Models
  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log(`Error creating model: ${e.message}`, true);
    state.studentModel = createBaselineModel(); // Fallback to avoid crash
  }

  // Create new optimizer (must be done AFTER models are created)
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  const baseRaw = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  const baseVis = baseRaw.clipByValue(0, 1);

  await tf.browser.toPixels(
    baseVis.squeeze(),
    document.getElementById("canvas-baseline"),
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student"),
  );

  baseRaw.dispose();
  studScores.dispose();
  studPred.dispose();
  baseVis.dispose();
}

// UI Helpers
function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText =
    `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

// Auto Train Logic
function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const tFlat = yTrue.reshape([256]);
    const pFlat = yPred.reshape([256]);

    // JS sort (как делал преподаватель в демо)
    const tArr = Array.from(tFlat.dataSync()).sort((a, b) => a - b);
    const pArr = Array.from(pFlat.dataSync()).sort((a, b) => a - b);

    const tSorted = tf.tensor1d(tArr);
    const pSorted = tf.tensor1d(pArr);

    return mse(tSorted, pSorted);
  });
}
// TODO-B: Custom Loss (Gradient Puzzle)
// Goal:
// 1. Allow pixel rearrangement (NOT position-locked like MSE)
// 2. Preserve color inventory (Histogram ≈ same)
// 3. Encourage smooth gradient structure
// 4. Encourage left-dark -> right-bright direction

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossSorted = sortedMSE(yTrue, yPred);  // conservation
    const lossSmooth = smoothness(yPred);        // TV loss
    const lossDir = directionX(yPred);           // direction

    return tf.addN([
      lossSorted.mul(1.0),   // НЕ 5.0 !!!
      lossSmooth.mul(3.0),   // усилить структуру
      lossDir.mul(1.0)       // усилить градиент
    ]);
  });
}
// Start
init();
