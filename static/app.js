const form = document.querySelector('#upload-form');
const errorMessage = document.querySelector('#error-message');
const resultsContainer = document.querySelector('#results-container');
const emptyState = document.querySelector('#empty-state');
const originalImage = document.querySelector('#original-image');
const overlayImage = document.querySelector('#overlay-image');
const rgbChart = document.querySelector('#rgb-chart');
const nailGrid = document.querySelector('#nail-grid');

function setError(message) {
  if (!message) {
    errorMessage.classList.add('hidden');
    errorMessage.textContent = '';
    return;
  }
  errorMessage.textContent = message;
  errorMessage.classList.remove('hidden');
}

function clearResults() {
  resultsContainer.classList.add('hidden');
  emptyState.classList.remove('hidden');
  originalImage.src = '';
  overlayImage.src = '';
  rgbChart.innerHTML = '';
  nailGrid.innerHTML = '';
}

function renderRgbGraph(graph) {
  rgbChart.innerHTML = '';
  const entries = [
    { label: 'Red', value: graph.red, className: 'red' },
    { label: 'Green', value: graph.green, className: 'green' },
    { label: 'Blue', value: graph.blue, className: 'blue' },
  ];
  entries.forEach((item) => {
    const bar = document.createElement('div');
    bar.className = 'rgb-bar';
    bar.innerHTML = `
      <div class="bar-track"><div class="bar-fill ${item.className}" style="height: ${item.value}%;"></div></div>
      <span class="bar-label">${item.label}</span>
    `;
    rgbChart.appendChild(bar);
  });
}

function renderNails(nails) {
  nailGrid.innerHTML = '';
  if (!Array.isArray(nails) || nails.length === 0) {
    nailGrid.innerHTML = '<div class="empty-state">No nail regions were detected.</div>';
    return;
  }
  nails.forEach((nail) => {
    const card = document.createElement('div');
    card.className = 'nail-card';
    card.innerHTML = `
      <img src="data:image/png;base64,${nail.image}" alt="Nail detail" />
      <div class="pill">${nail.color}</div>
      <div class="hint">RGB ${nail.rgb}</div>
      <div>${nail.conclusion}</div>
    `;
    nailGrid.appendChild(card);
  });
}

async function submitImage(event) {
  event.preventDefault();
  setError('');
  clearResults();

  const input = document.querySelector('#image-input');
  if (!input.files || input.files.length === 0) {
    setError('Please choose an image file.');
    return;
  }

  const formData = new FormData();
  formData.append('image', input.files[0]);

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      setError(data.error || 'Unable to process the image.');
      return;
    }

    originalImage.src = `data:image/png;base64,${data.result_image}`;
    overlayImage.src = `data:image/png;base64,${data.overlay_image}`;
    renderRgbGraph(data.analysis.rgb_graph);
    renderNails(data.analysis.nails);
    resultsContainer.classList.remove('hidden');
    emptyState.classList.add('hidden');
  } catch (error) {
    setError(error.message || 'Failed to analyze the image.');
  }
}

form.addEventListener('submit', submitImage);
