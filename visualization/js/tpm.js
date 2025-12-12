// TPM (Transition Probability Matrix) utilities
export function generateTPM(size) {
    return Array(size).fill(0).map(() =>
        Array(size).fill(0).map(() => Math.random())
    );
}

export function updateTPMDisplay(tpmData) {
    const tpmMatrix = document.getElementById('tpmMatrix');
    const emptyState = document.getElementById('emptyState');
    
    if (tpmData.length === 0) {
        tpmMatrix.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }

    tpmMatrix.style.display = 'flex';
    emptyState.style.display = 'none';
    tpmMatrix.innerHTML = '';

    tpmData.forEach((row) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'tpm-row';

        row.forEach((val) => {
            const cell = document.createElement('div');
            cell.className = 'tpm-cell';
            const opacity = val * 0.8;
            cell.style.backgroundColor = `rgba(99, 102, 241, ${opacity})`;
            cell.style.color = val > 0.5 ? '#fff' : '#9ca3af';
            cell.textContent = val.toFixed(2);
            cell.title = val.toFixed(4);
            rowDiv.appendChild(cell);
        });

        tpmMatrix.appendChild(rowDiv);
    });
}