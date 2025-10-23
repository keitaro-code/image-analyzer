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
const imagePreviewSection = document.getElementById('image-preview');
const imagePreviewImg = imagePreviewSection?.querySelector('img');
const imagePreviewCaption = imagePreviewSection?.querySelector('.image-filename');
const clarificationSection = document.getElementById('clarification');
const questionText = clarificationSection?.querySelector('.question-text');
const questionContext = clarificationSection?.querySelector('.question-context');
const answerPreview = clarificationSection?.querySelector('.answer-preview');
const answerForm = document.getElementById('answer-form');
const answerInput = document.getElementById('answer-input');
const answerButton = document.getElementById('answer-submit');
const resultSection = document.getElementById('result');
const candidateEl = resultSection.querySelector('.candidate');
const confidenceFillEl = resultSection.querySelector('.confidence-fill');
const confidencePercentEl = resultSection.querySelector('.confidence-percent');
const confidenceLabelEl = resultSection.querySelector('.confidence-label');
const reasonEl = resultSection.querySelector('.reason');
const retryButton = document.getElementById('retry-button');

let pollingTimer = null;
const defaultFileLabel = '画像を選択';
let currentTaskId = null;
let currentPreviewUrl = null;
let lastLoggedNotes = '';
let lastLoggedError = '';
let lastLoggedStep = '';
let repeatedStepCount = 0;

const showImagePreview = (file) => {
  if (!file || !imagePreviewSection || !imagePreviewImg || !imagePreviewCaption) {
    return;
  }
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
  }
  currentPreviewUrl = URL.createObjectURL(file);
  imagePreviewImg.src = currentPreviewUrl;
  imagePreviewImg.alt = `${file.name} のプレビュー`;
  imagePreviewCaption.textContent = file.name;
  imagePreviewSection.classList.remove('is-hidden');
};

const resetUI = (options = {}) => {
  const { preservePreview = false, preserveFilename = false } = options;
  statusEl.textContent = '';
  progressSection.hidden = true;
  progressBar.value = 0;
  stepText.textContent = '';
  notesContainer.hidden = true;
  notesPre.textContent = '';
  lastLoggedNotes = '';
  lastLoggedError = '';
  lastLoggedStep = '';
  repeatedStepCount = 0;
  if (!preservePreview && imagePreviewSection) {
    imagePreviewSection.classList.add('is-hidden');
    if (imagePreviewImg) imagePreviewImg.src = '';
    if (imagePreviewCaption) imagePreviewCaption.textContent = '';
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = null;
    }
  }
  if (clarificationSection) {
    clarificationSection.hidden = true;
    clarificationSection.style.display = 'none';
    if (questionText) questionText.textContent = '';
    if (questionContext) questionContext.textContent = '';
    if (answerPreview) {
      answerPreview.textContent = '';
      answerPreview.style.display = 'none';
    }
    if (answerInput) answerInput.value = '';
    if (answerInput) answerInput.readOnly = false;
    if (answerButton) answerButton.disabled = false;
    delete clarificationSection.dataset.state;
    delete clarificationSection.dataset.question;
    delete clarificationSection.dataset.context;
    delete clarificationSection.dataset.answer;
  }
  resultSection.classList.add('is-hidden');
  candidateEl.textContent = '';
  candidateEl.className = 'candidate';
  if (confidenceFillEl) {
    confidenceFillEl.style.width = '0%';
    confidenceFillEl.classList.remove('level-low', 'level-mid', 'level-high');
  }
  if (confidencePercentEl) confidencePercentEl.textContent = '--%';
  if (confidenceLabelEl) confidenceLabelEl.textContent = '---';
  reasonEl.textContent = '';
  if (!preserveFilename) {
    fileLabelText.textContent = defaultFileLabel;
  }
};

const formatConfidence = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return { percent: null, level: 'low', label: '---' };
  }
  const normalized = Math.min(Math.max(value, 0), 1);
  const adjusted = 0.4 + normalized * 0.45; // 40%〜85% の範囲で表示
  const percent = Math.round(adjusted * 100);

  let level = 'low';
  let label = '参考程度';
  if (percent >= 75) {
    level = 'high';
    label = 'かなり確信あり';
  } else if (percent >= 60) {
    level = 'mid';
    label = 'まずまず';
  } else if (percent >= 50) {
    level = 'mid';
    label = '判断保留';
  }

  return { percent, level, label };
};

const stopPolling = () => {
  if (pollingTimer) {
    clearInterval(pollingTimer);
    pollingTimer = null;
  }
};

const updateStatusUI = (payload) => {
  const {
    task_id: taskId,
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
    if (notes !== lastLoggedNotes) {
      console.info(`[Task ${taskId ?? 'unknown'}] Notes updated:\n${notes}`);
      lastLoggedNotes = notes;
    }
  }

  if (clarificationSection) {
    const previousQuestion = clarificationSection.dataset.question || '';
    const storedAnswer = clarificationSection.dataset.answer || '';
    const hasSubmitted = clarificationSection.dataset.state === 'submitted';
    const stillProcessing = status !== 'completed' && status !== 'failed';

    const showSubmittedPreview = (answerText) => {
      clarificationSection.hidden = false;
      clarificationSection.style.display = 'flex';
      if (questionText) questionText.textContent = '提出いただいた情報';
      if (questionContext) {
        questionContext.textContent = '';
        questionContext.style.display = 'none';
      }
      if (answerInput) {
        answerInput.readOnly = true;
        answerInput.style.display = 'none';
      }
      if (answerButton) {
        answerButton.disabled = true;
        answerButton.style.display = 'none';
      }
      if (answerPreview) {
        answerPreview.textContent = answerText;
        answerPreview.style.display = 'block';
      }
      uploadButton.disabled = true;
    };

    if (awaiting_answer) {
      clarificationSection.hidden = false;
      clarificationSection.style.display = 'flex';
      const newQuestion = question ?? '追加の情報を教えてください。';
      const sameQuestionAsBefore = previousQuestion === newQuestion;

      if (hasSubmitted && sameQuestionAsBefore && storedAnswer) {
        // 送信済みだがモデルからまだ応答が返っていない。プレビューのまま維持。
        showSubmittedPreview(storedAnswer);
        return;
      }

      const mustResetInput = !sameQuestionAsBefore;
      if (questionText) questionText.textContent = newQuestion;
      if (questionContext) {
        questionContext.textContent = context ?? '';
        questionContext.style.display = context ? 'block' : 'none';
      }
      if (answerButton) {
        answerButton.disabled = false;
        answerButton.style.display = '';
      }
      if (answerInput) {
        answerInput.readOnly = false;
        answerInput.style.display = '';
        if (mustResetInput) {
          answerInput.value = '';
        }
      }
      if (answerPreview) {
        answerPreview.textContent = '';
        answerPreview.style.display = 'none';
      }
      clarificationSection.dataset.question = newQuestion;
      clarificationSection.dataset.context = context ?? '';
      if (!sameQuestionAsBefore) {
        delete clarificationSection.dataset.answer;
      }
      clarificationSection.dataset.state = 'awaiting';
      uploadButton.disabled = true;
      statusEl.textContent = '追加情報を入力してください。';
    } else if (hasSubmitted && stillProcessing) {
      showSubmittedPreview(storedAnswer);
    } else {
      clarificationSection.hidden = true;
      clarificationSection.style.display = 'none';
      if (questionText) questionText.textContent = '';
      if (questionContext) {
        questionContext.textContent = '';
        questionContext.style.display = 'none';
      }
      if (answerPreview) {
        answerPreview.textContent = '';
        answerPreview.style.display = 'none';
      }
      if (answerInput) {
        answerInput.value = '';
        answerInput.readOnly = false;
        answerInput.style.display = '';
      }
      if (answerButton) {
        answerButton.disabled = false;
        answerButton.style.display = '';
      }
      delete clarificationSection.dataset.state;
      delete clarificationSection.dataset.question;
      delete clarificationSection.dataset.context;
      delete clarificationSection.dataset.answer;
    }
  }

  if (error) {
    statusEl.textContent = `エラー: ${error}`;
    if (error !== lastLoggedError) {
      console.error(`[Task ${taskId ?? 'unknown'}] Error: ${error}`);
      lastLoggedError = error;
    }
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
    if (confidenceFillEl) {
      confidenceFillEl.style.width = conf.percent !== null ? `${conf.percent}%` : '0%';
      confidenceFillEl.classList.remove('level-low', 'level-mid', 'level-high');
      const fillLevelClass = conf.level === 'high' ? 'level-high' : conf.level === 'mid' ? 'level-mid' : 'level-low';
      confidenceFillEl.classList.add(fillLevelClass);
    }
    if (confidencePercentEl) {
      confidencePercentEl.textContent = conf.percent !== null ? `${conf.percent}%` : '--%';
    }
    if (confidenceLabelEl) {
      confidenceLabelEl.textContent = conf.label;
    }
    reasonEl.textContent = reason ?? '---';
  }

  if (status === 'completed' || status === 'failed') {
    stopPolling();
    uploadButton.disabled = false;
  }

  if (status === 'processing') {
    const currentStep = step || '(ステップ未設定)';
    if (currentStep === lastLoggedStep) {
      repeatedStepCount += 1;
    } else {
      lastLoggedStep = currentStep;
      repeatedStepCount = 1;
    }
    if (repeatedStepCount % 10 === 0) {
      console.warn(
        `[Task ${taskId ?? 'unknown'}] Still processing at step "${currentStep}" after ${repeatedStepCount} updates.`,
      );
    }
  } else {
    lastLoggedStep = '';
    repeatedStepCount = 0;
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

  resetUI({ preservePreview: true, preserveFilename: true });
  uploadButton.disabled = true;
  statusEl.textContent = '画像を送信しています...';
  currentTaskId = null;

  progressSection.hidden = false;
  progressBar.value = 5;
  stepText.textContent = '画像を送信中';

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
      answerButton.style.display = 'none';
    }
    if (answerInput) {
      answerInput.readOnly = true;
      answerInput.style.display = 'none';
    }
    if (clarificationSection) {
      clarificationSection.dataset.state = 'submitted';
      clarificationSection.dataset.answer = answer;
    }
    if (questionText) questionText.textContent = '提出いただいた情報';
    if (questionContext) {
      questionContext.textContent = '';
      questionContext.style.display = 'none';
    }
    if (answerPreview) {
      answerPreview.textContent = answer;
      answerPreview.style.display = 'block';
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
      updateStatusUI(payload);
      if (!pollingTimer) {
        pollStatus(currentTaskId);
      }
    } catch (error) {
      console.error(error);
      statusEl.textContent = '回答の送信に失敗しました。もう一度お試しください。';
      if (answerButton) {
        answerButton.disabled = false;
        answerButton.style.display = '';
      }
      if (answerInput) {
        answerInput.readOnly = false;
        answerInput.style.display = '';
      }
      if (clarificationSection) {
        clarificationSection.dataset.state = 'awaiting';
        if (questionText) {
          questionText.textContent = clarificationSection.dataset.question || '追加の情報を教えてください。';
        }
        if (questionContext) {
          const savedContext = clarificationSection.dataset.context || '';
          questionContext.textContent = savedContext;
          questionContext.style.display = savedContext ? 'block' : 'none';
        }
        if (answerPreview) {
          answerPreview.textContent = '';
          answerPreview.style.display = 'none';
        }
        delete clarificationSection.dataset.answer;
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
    showImagePreview(imageInput.files[0]);
  } else {
    fileLabelText.textContent = defaultFileLabel;
    resetUI();
  }
});
