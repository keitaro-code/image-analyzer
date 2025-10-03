const API_BASE_URL = 'https://image-analyzer-5c3x.onrender.com';

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
const clarificationSection = document.getElementById('clarification');
const questionText = clarificationSection?.querySelector('.question-text');
const questionContext = clarificationSection?.querySelector('.question-context');
const answerForm = document.getElementById('answer-form');
const answerInput = document.getElementById('answer-input');
const answerButton = document.getElementById('answer-submit');
const resultSection = document.getElementById('result');
const candidateEl = resultSection.querySelector('.candidate');
const confidenceEl = resultSection.querySelector('.confidence');
const reasonEl = resultSection.querySelector('.reason');
const retryButton = document.getElementById('retry-button');

let pollingTimer = null;
const defaultFileLabel = '画像を選択';
let currentTaskId = null;

const resetUI = () => {
  statusEl.textContent = '';
  progressSection.hidden = true;
  progressBar.value = 0;
  stepText.textContent = '';
  notesContainer.hidden = true;
  notesPre.textContent = '';
  if (clarificationSection) {
    clarificationSection.hidden = true;
    if (questionText) questionText.textContent = '';
    if (questionContext) questionContext.textContent = '';
    if (answerInput) answerInput.value = '';
    if (answerButton) answerButton.disabled = false;
  }
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
    question,
    question_context: context,
    awaiting_answer,
  } = payload;

  progressSection.hidden = false;
  progressBar.value = progress ?? 0;
  stepText.textContent = step ?? '進行中';

  if (notes) {
    notesContainer.hidden = false;
    notesPre.textContent = notes.trim();
    notesPre.scrollTop = notesPre.scrollHeight;
  }

  if (awaiting_answer && clarificationSection) {
    clarificationSection.hidden = false;
    if (questionText) {
      questionText.textContent = question ?? '追加の情報を教えてください。';
    }
    if (questionContext) {
      questionContext.textContent = context ?? '';
      questionContext.style.display = context ? 'block' : 'none';
    }
    if (answerButton) {
      answerButton.disabled = false;
    }
    uploadButton.disabled = true;
    statusEl.textContent = '追加情報を入力してください。';
  } else if (clarificationSection) {
    clarificationSection.hidden = true;
    if (questionText) questionText.textContent = '';
    if (questionContext) questionContext.textContent = '';
    if (answerInput) answerInput.value = '';
  }

  if (error) {
    statusEl.textContent = `エラー: ${error}`;
  } else {
    if (!awaiting_answer && status !== 'completed') {
      statusEl.textContent = '解析を実行中です...';
    } else if (status === 'completed') {
      statusEl.textContent = '解析が完了しました';
    }
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
      const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
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
  currentTaskId = null;

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`アップロードに失敗しました (${response.status})`);
    }

    const payload = await response.json();
    updateStatusUI(payload);

    if (payload.task_id) {
      currentTaskId = payload.task_id;
      pollStatus(payload.task_id);
    }
  } catch (error) {
    console.error(error);
    statusEl.textContent = 'アップロードに失敗しました。再度お試しください。';
    uploadButton.disabled = false;
  }
});

if (answerForm) {
  answerForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!currentTaskId) {
      statusEl.textContent = 'タスクが見つかりません。再度解析を実行してください。';
      return;
    }

    const answer = answerInput.value.trim();
    if (!answer) {
      statusEl.textContent = '回答内容を入力してください。';
      return;
    }

    if (answerButton) {
      answerButton.disabled = true;
    }
    statusEl.textContent = '回答を送信しています...';

    try {
      const response = await fetch(`${API_BASE_URL}/answer/${currentTaskId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ answer }),
      });

      if (!response.ok) {
        throw new Error(`回答送信に失敗しました (${response.status})`);
      }

      const payload = await response.json();
      answerInput.value = '';
      updateStatusUI(payload);
      if (!pollingTimer) {
        pollStatus(currentTaskId);
      }
    } catch (error) {
      console.error(error);
      statusEl.textContent = '回答の送信に失敗しました。もう一度お試しください。';
      if (answerButton) {
        answerButton.disabled = false;
      }
    }
  });
}

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
