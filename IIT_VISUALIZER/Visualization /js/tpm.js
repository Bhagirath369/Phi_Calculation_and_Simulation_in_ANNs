/**
 * TPM and MIP Matrix utilities
 */

export function generateTPM(size) {
  return Array(size)
    .fill(0)
    .map(() =>
      Array(size)
        .fill(0)
        .map(() => Math.random()),
    );
}

export function generateMIPMatrix(size) {
  // Generate MIP matrix with slightly different probabilities
  // MIP represents minimum information partition
  return Array(size)
    .fill(0)
    .map(
      () =>
        Array(size)
          .fill(0)
          .map(() => Math.random() * 0.8), // Slightly lower probabilities
    );
}

export function updateTPMDisplay(tpmData) {
  const tpmMatrix = document.getElementById("tpmMatrix");
  const emptyState = document.getElementById("emptyState");

  if (tpmData.length === 0) {
    tpmMatrix.style.display = "none";
    emptyState.style.display = "block";
    return;
  }

  tpmMatrix.style.display = "flex";
  emptyState.style.display = "none";
  tpmMatrix.innerHTML = "";

  // Add title
  const title = document.createElement("div");
  title.className = "matrix-title";
  title.textContent = "Full System TPM";
  tpmMatrix.appendChild(title);

  // Add matrix grid
  const matrixGrid = document.createElement("div");
  matrixGrid.className = "matrix-grid";

  tpmData.forEach((row) => {
    const rowDiv = document.createElement("div");
    rowDiv.className = "tpm-row";

    row.forEach((val) => {
      const cell = document.createElement("div");
      cell.className = "tpm-cell";
      const opacity = val * 0.8;
      cell.style.backgroundColor = `rgba(99, 102, 241, ${opacity})`;
      cell.style.color = val > 0.5 ? "#fff" : "#9ca3af";
      cell.textContent = val.toFixed(2);
      cell.title = `Probability: ${val.toFixed(4)}`;
      rowDiv.appendChild(cell);
    });

    matrixGrid.appendChild(rowDiv);
  });

  tpmMatrix.appendChild(matrixGrid);
}

export function updateMIPDisplay(mipData) {
  const mipMatrix = document.getElementById("mipMatrix");

  if (mipData.length === 0) {
    mipMatrix.style.display = "none";
    return;
  }

  mipMatrix.style.display = "flex";
  mipMatrix.innerHTML = "";

  // Add title
  const title = document.createElement("div");
  title.className = "matrix-title";
  title.textContent = "MIP (Minimum Information Partition)";
  mipMatrix.appendChild(title);

  // Add matrix grid
  const matrixGrid = document.createElement("div");
  matrixGrid.className = "matrix-grid";

  mipData.forEach((row) => {
    const rowDiv = document.createElement("div");
    rowDiv.className = "tpm-row";

    row.forEach((val) => {
      const cell = document.createElement("div");
      cell.className = "tpm-cell mip-cell";
      const opacity = val * 0.8;
      cell.style.backgroundColor = `rgba(139, 92, 246, ${opacity})`;
      cell.style.color = val > 0.5 ? "#fff" : "#9ca3af";
      cell.textContent = val.toFixed(2);
      cell.title = `MIP Probability: ${val.toFixed(4)}`;
      rowDiv.appendChild(cell);
    });

    matrixGrid.appendChild(rowDiv);
  });

  mipMatrix.appendChild(matrixGrid);

  // Add partition indicator
  const partitionInfo = document.createElement("div");
  partitionInfo.className = "partition-info";
  partitionInfo.innerHTML = `
        <div class="partition-label">
            <span class="partition-dot partition-a"></span>
            <span>Subsystem A</span>
        </div>
        <div class="partition-label">
            <span class="partition-dot partition-b"></span>
            <span>Subsystem B</span>
        </div>
    `;
  mipMatrix.appendChild(partitionInfo);
}
