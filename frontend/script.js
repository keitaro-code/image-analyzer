const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const uploadButton = document.getElementById('upload-button');
const fileLabelText = document.querySelector('.file-label-text');
const statusEl = document.getElementById('status');
const progressSection = document.getElementById('progress');
const progressBar = progressSection.querySelector('progress');
const stepText = progressSection.querySelector('.step');
const notesContainer = document.getElementById('notes');
const notesPre = notesContainer.querySelector('pre');
const resultSection = document.getElementById('result');
const candidateEl = resultSection.querySelector('.candidate');
const confidenceEl = resultSection.querySelector('.confidence');
const reasonEl = resultSection.querySelector('.reason');
const retryButton = document.getElementById('retry-button');

let pollingTimer = null;
const defaultFileLabel = '画像を選択';

const resetUI = () => {
  statusEl.textContent = '';
  progressSection.hidden = true;
  progressBar.value = 0;
  stepText.textContent = '';
  notesContainer.hidden = true;
  notesPre.textContent = '';
  resultSection.classList.add('is-hidden');
  candidateEl.textContent = '';
  candidateEl.className = 'candidate';
  confidenceEl.textContent = '';
  confidenceEl.className = 'confidence';
  reasonEl.textContent = '';
  fileLabelText.textContent = defaultFileLabel;
};

const formatConfidence = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return { text: '---', level: 'low' };
  }
  const percent = Math.round(value * 100);
  let level = 'low';
  if (percent >= 80) {
    level = 'high';
  } else if (percent >= 50) {
    level = 'mid';
  }
  return { text: `${percent}%`, level };
};

const stopPolling = () => {
  if (pollingTimer) {
    clearInterval(pollingTimer);
    pollingTimer = null;
  }
};

const updateStatusUI = (payload) => {
  const {
    step,
    progress,
    candidate,
    confidence,
    reason,
    status,
    error,
    notes,
  } = payload;

  progressSection.hidden = false;
  progressBar.value = progress ?? 0;
  stepText.textContent = step ?? '進行中';

  if (notes) {
    notesContainer.hidden = false;
    notesPre.textContent = notes.trim();
    notesPre.scrollTop = notesPre.scrollHeight;
  }

  if (error) {
    statusEl.textContent = `エラー: ${error}`;
  } else {
    statusEl.textContent = status === 'completed' ? '解析が完了しました' : '解析を実行中です...';
  }

  if (candidate && status === 'completed') {
    resultSection.classList.remove('is-hidden');
    candidateEl.textContent = candidate;
    candidateEl.className = 'candidate highlight-text';
    const conf = formatConfidence(confidence);
    confidenceEl.textContent = '';
    confidenceEl.className = `confidence confidence-pill ${conf.level}`;
    confidenceEl.textContent = conf.text;
    reasonEl.textContent = reason ?? '---';
  }

  if (status === 'completed' || status === 'failed') {
    stopPolling();
    uploadButton.disabled = false;
  }
};

const pollStatus = (taskId) => {
  stopPolling();
  pollingTimer = setInterval(async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/status/${taskId}`);
      if (!response.ok) {
        throw new Error(`ステータス取得に失敗しました (${response.status})`);
      }
      const payload = await response.json();
      updateStatusUI(payload);
      if (payload.status === 'completed' || payload.status === 'failed') {
        stopPolling();
      }
    } catch (error) {
      console.error(error);
      statusEl.textContent = 'ステータス更新に失敗しました。ネットワークを確認して再試行してください。';
      stopPolling();
    }
  }, 2000);
};

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!imageInput.files || imageInput.files.length === 0) {
    statusEl.textContent = '解析する画像を選択してください。';
    return;
  }

  resetUI();
  uploadButton.disabled = true;
  statusEl.textContent = '画像を送信しています...';

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  try {
    const response = await fetch('http://127.0.0.1:8000/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`アップロードに失敗しました (${response.status})`);
    }

    const payload = await response.json();
    updateStatusUI(payload);

    if (payload.task_id) {
      pollStatus(payload.task_id);
    }
  } catch (error) {
    console.error(error);
    statusEl.textContent = 'アップロードに失敗しました。再度お試しください。';
    uploadButton.disabled = false;
  }
});

retryButton.addEventListener('click', () => {
  stopPolling();
  resetUI();
  imageInput.value = '';
  uploadButton.disabled = false;
  statusEl.textContent = '再度画像を選択してください。';
});

imageInput.addEventListener('change', () => {
  if (imageInput.files && imageInput.files.length > 0) {
    fileLabelText.textContent = imageInput.files[0].name;
  } else {
    fileLabelText.textContent = defaultFileLabel;
  }
});
