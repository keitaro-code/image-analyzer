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
const timelineContainer = notesContainer?.querySelector('.timeline');
const notesFallbackPre = notesContainer?.querySelector('.notes-fallback');
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
const answerAttachmentInput = document.getElementById('answer-attachments');
const answerAttachmentList = clarificationSection?.querySelector('.attachment-list');
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
let renderedTimelineIds = new Set();
let answerAttachmentFiles = [];
const MAX_ATTACHMENT_COUNT = 3;

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

const clearAnswerAttachments = () => {
  answerAttachmentFiles.forEach((item) => {
    if (item.url) {
      URL.revokeObjectURL(item.url);
    }
  });
  answerAttachmentFiles = [];
  if (answerAttachmentList) {
    answerAttachmentList.innerHTML = '';
  }
  if (answerAttachmentInput) {
    answerAttachmentInput.value = '';
    answerAttachmentInput.disabled = false;
    answerAttachmentInput.style.display = '';
  }
  const attachmentLabel = clarificationSection?.querySelector('.attachment-label span');
  if (attachmentLabel) {
    attachmentLabel.textContent = '参考画像を添付（最大3枚）';
  }
};

const renderAnswerAttachments = ({ readonly = false } = {}) => {
  if (!answerAttachmentList) {
    return;
  }
  answerAttachmentList.innerHTML = '';
  answerAttachmentFiles.forEach((item) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'attachment-item';

    const thumbnail = document.createElement('img');
    thumbnail.src = item.url;
    thumbnail.alt = `${item.file.name} のプレビュー`;
    wrapper.appendChild(thumbnail);

    const meta = document.createElement('div');
    meta.className = 'attachment-meta';

    const nameEl = document.createElement('span');
    nameEl.className = 'attachment-name';
    nameEl.textContent = item.file.name;
    meta.appendChild(nameEl);

    const sizeEl = document.createElement('span');
    sizeEl.className = 'attachment-size';
    const sizeKB = Math.round(item.file.size / 1024);
    sizeEl.textContent = `${sizeKB} KB`;
    meta.appendChild(sizeEl);

    wrapper.appendChild(meta);

    if (!readonly) {
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'attachment-remove';
      removeBtn.textContent = '削除';
      removeBtn.addEventListener('click', () => {
        answerAttachmentFiles = answerAttachmentFiles.filter((candidate) => candidate.id !== item.id);
        if (item.url) {
          URL.revokeObjectURL(item.url);
        }
        renderAnswerAttachments();
        if (answerAttachmentInput) {
          answerAttachmentInput.disabled = answerAttachmentFiles.length >= MAX_ATTACHMENT_COUNT;
          if (!answerAttachmentInput.disabled) {
            answerAttachmentInput.style.display = '';
          }
        }
      });
      wrapper.appendChild(removeBtn);
    }

    answerAttachmentList.appendChild(wrapper);
  });

  if (answerAttachmentInput) {
    if (readonly) {
      answerAttachmentInput.disabled = true;
      answerAttachmentInput.style.display = 'none';
    } else {
      answerAttachmentInput.disabled = answerAttachmentFiles.length >= MAX_ATTACHMENT_COUNT;
      answerAttachmentInput.style.display = answerAttachmentInput.disabled ? 'none' : '';
    }
  }

  const attachmentLabel = clarificationSection?.querySelector('.attachment-label span');
  if (attachmentLabel) {
    if (readonly) {
      attachmentLabel.textContent = answerAttachmentFiles.length
        ? '添付された画像'
        : '画像は添付されていません';
    } else {
      const remaining = MAX_ATTACHMENT_COUNT - answerAttachmentFiles.length;
      attachmentLabel.textContent = remaining === MAX_ATTACHMENT_COUNT
        ? '参考画像を添付（最大3枚）'
        : `参考画像を添付（残り ${remaining} 枚）`;
    }
  }
};

const addAnswerAttachments = (fileList) => {
  if (!fileList || fileList.length === 0) {
    return;
  }
  let updated = false;
  Array.from(fileList).forEach((file) => {
    if (!(file instanceof File)) {
      return;
    }
    if (!file.type || !file.type.startsWith('image/')) {
      statusEl.textContent = '画像ファイルのみ添付できます。';
      return;
    }
    if (answerAttachmentFiles.length >= MAX_ATTACHMENT_COUNT) {
      statusEl.textContent = `画像は最大 ${MAX_ATTACHMENT_COUNT} 枚まで添付できます。`;
      return;
    }
    const id = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const url = URL.createObjectURL(file);
    answerAttachmentFiles.push({ id, file, url });
    updated = true;
  });

  if (answerAttachmentInput) {
    answerAttachmentInput.value = '';
  }

  if (updated) {
    renderAnswerAttachments();
    statusEl.textContent = '';
  }
};
const formatTimelineTimestamp = (isoString) => {
  if (!isoString) {
    return '';
  }
  const parsed = new Date(isoString);
  if (Number.isNaN(parsed.getTime())) {
    return '';
  }
  return parsed.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });
};

const getTimelineTypeClass = (type) => {
  if (!type) {
    return 'timeline-event--default';
  }
  return `timeline-event--${type}`;
};

const appendTimelineEvents = (events = [], taskId) => {
  if (!timelineContainer) {
    return;
  }
  let appended = false;
  events.forEach((event) => {
    if (!event || !event.id || renderedTimelineIds.has(event.id)) {
      return;
    }
    const container = document.createElement('article');
    container.className = `timeline-event ${getTimelineTypeClass(event.type)}`;

    const header = document.createElement('div');
    header.className = 'timeline-event-header';

    const titleEl = document.createElement('h4');
    titleEl.textContent = event.title || '進捗更新';
    header.appendChild(titleEl);

    const timeLabel = formatTimelineTimestamp(event.timestamp);
    if (timeLabel) {
      const timeEl = document.createElement('time');
      timeEl.className = 'timeline-event-time';
      timeEl.dateTime = event.timestamp;
      timeEl.textContent = timeLabel;
      header.appendChild(timeEl);
    }

    container.appendChild(header);

    if (event.body) {
      const bodyEl = document.createElement('p');
      bodyEl.className = 'timeline-event-body';
      bodyEl.textContent = event.body;
      container.appendChild(bodyEl);
    }

    if (Array.isArray(event.items) && event.items.length > 0) {
      const listEl = document.createElement('ul');
      listEl.className = 'timeline-event-items';
      event.items.forEach((item) => {
        if (!item) {
          return;
        }
        const li = document.createElement('li');
        li.textContent = item;
        listEl.appendChild(li);
      });
      if (listEl.childElementCount > 0) {
        container.appendChild(listEl);
      }
    }

    const context = event.meta?.context;
    if (context && typeof context === 'string' && context.trim()) {
      const contextEl = document.createElement('p');
      contextEl.className = 'timeline-event-context';
      contextEl.textContent = context.trim();
      container.appendChild(contextEl);
    }

    timelineContainer.appendChild(container);
    if (taskId) {
      console.info(`[Task ${taskId}] Timeline update: ${event.title || event.type || event.id}`);
    }
    renderedTimelineIds.add(event.id);
    appended = true;
  });

  if (appended) {
    timelineContainer.classList.remove('is-hidden');
    timelineContainer.scrollTop = timelineContainer.scrollHeight;
  }
};

const formatReasonText = (text) => {
  if (!text) {
    return '---';
  }
  let formatted = text;
  formatted = formatted.replace(/\s*-\s*/g, '\n- ');
  formatted = formatted.replace(/(\d+\.)\s*/g, '\n$1 ');
  formatted = formatted.replace(/\n{2,}/g, '\n');
  formatted = formatted.replace(/^\n+/, '');
  return formatted.trim();
};

const resetUI = (options = {}) => {
  const { preservePreview = false, preserveFilename = false } = options;
  statusEl.textContent = '';
  progressSection.hidden = true;
  progressBar.value = 0;
  stepText.textContent = '';
  notesContainer.hidden = true;
  if (timelineContainer) {
    timelineContainer.innerHTML = '';
    timelineContainer.classList.remove('is-hidden');
  }
  if (notesFallbackPre) {
    notesFallbackPre.textContent = '';
    notesFallbackPre.classList.remove('is-hidden');
  }
  renderedTimelineIds = new Set();
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
    clearAnswerAttachments();
    delete clarificationSection.dataset.state;
    delete clarificationSection.dataset.question;
    delete clarificationSection.dataset.context;
    delete clarificationSection.dataset.answer;
    delete clarificationSection.dataset.attachments;
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

  const timelineEvents = Array.isArray(payload.timeline) ? payload.timeline : [];
  if (timelineEvents.length > 0) {
    notesContainer.hidden = false;
    appendTimelineEvents(timelineEvents, taskId);
    if (notesFallbackPre) {
      notesFallbackPre.classList.add('is-hidden');
    }
    if (typeof notes === 'string') {
      lastLoggedNotes = notes;
    }
  } else if (notes) {
    notesContainer.hidden = false;
    if (timelineContainer) {
      timelineContainer.classList.add('is-hidden');
    }
    if (notesFallbackPre) {
      notesFallbackPre.classList.remove('is-hidden');
      notesFallbackPre.textContent = notes.trim();
      notesFallbackPre.scrollTop = notesFallbackPre.scrollHeight;
    }
    if (notes !== lastLoggedNotes) {
      console.info(`[Task ${taskId ?? 'unknown'}] Notes updated:\n${notes}`);
      lastLoggedNotes = notes;
    }
  } else if (timelineContainer && timelineContainer.childElementCount === 0) {
    notesContainer.hidden = true;
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
      renderAnswerAttachments({ readonly: true });
      if (answerPreview) {
        const attachmentNames = (() => {
          try {
            return JSON.parse(clarificationSection.dataset.attachments || '[]');
          } catch (error) {
            return [];
          }
        })();
        if (answerText) {
          answerPreview.textContent = answerText;
          answerPreview.style.display = 'block';
        } else if (attachmentNames.length > 0) {
          answerPreview.textContent = '添付した画像をご確認ください。';
          answerPreview.style.display = 'block';
        } else {
          answerPreview.textContent = '';
          answerPreview.style.display = 'none';
        }
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
      if (mustResetInput) {
        clearAnswerAttachments();
      }
      renderAnswerAttachments({ readonly: false });
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
      clearAnswerAttachments();
      delete clarificationSection.dataset.state;
      delete clarificationSection.dataset.question;
      delete clarificationSection.dataset.context;
      delete clarificationSection.dataset.answer;
      delete clarificationSection.dataset.attachments;
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
    reasonEl.textContent = formatReasonText(reason);
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
    const hasAttachments = answerAttachmentFiles.length > 0;
    if (!answer && !hasAttachments) {
      statusEl.textContent = 'テキストか画像のいずれかを入力してください。';
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
    renderAnswerAttachments({ readonly: true });
    if (clarificationSection) {
      clarificationSection.dataset.state = 'submitted';
      clarificationSection.dataset.answer = answer;
      clarificationSection.dataset.attachments = JSON.stringify(
        answerAttachmentFiles.map((item) => item.file.name),
      );
    }
    if (questionText) questionText.textContent = '提出いただいた情報';
    if (questionContext) {
      questionContext.textContent = '';
      questionContext.style.display = 'none';
    }
    if (answerPreview) {
      if (answer) {
        answerPreview.textContent = answer;
        answerPreview.style.display = 'block';
      } else if (hasAttachments) {
        answerPreview.textContent = '画像を添付しました。';
        answerPreview.style.display = 'block';
      } else {
        answerPreview.textContent = '';
        answerPreview.style.display = 'none';
      }
    }
    statusEl.textContent = '回答を送信しています...';

    try {
      const formData = new FormData();
      if (answer) {
        formData.append('answer', answer);
      }
      answerAttachmentFiles.forEach((item) => {
        if (item.file) {
          formData.append('images', item.file, item.file.name);
        }
      });

      const response = await fetch(`${API_BASE_URL}/answer/${currentTaskId}`, {
        method: 'POST',
        body: formData,
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
      renderAnswerAttachments({ readonly: false });
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
        delete clarificationSection.dataset.attachments;
      }
    }
  });
}

if (answerAttachmentInput) {
  answerAttachmentInput.addEventListener('change', (event) => {
    const { files } = event.target;
    addAnswerAttachments(files);
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
