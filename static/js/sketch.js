const canvas = document.getElementById("sketchCanvas");
const ctx = canvas.getContext("2d");
const categoryList = document.getElementById("categoryList");
const thumbnailPane = document.getElementById("thumbnailPane");
const resetButton = document.getElementById("resetCanvas");
const saveButton = document.getElementById("saveSketch");

let partsByCategory = {};
let layers = [];
let activeLayerIndex = -1;
let dragOffset = { x: 0, y: 0 };

async function loadParts() {
  const response = await fetch("/api/parts");
  partsByCategory = await response.json();

  const firstCategory = Object.keys(partsByCategory)[0];
  if (firstCategory) {
    renderCategoryButtons();
    renderThumbnails(firstCategory);
  }
}

function renderCategoryButtons() {
  categoryList.innerHTML = "";
  Object.keys(partsByCategory).forEach((category) => {
    const wrapper = document.createElement("li");
    const button = document.createElement("button");
    button.className = "w-full text-left px-3 py-2 rounded bg-slate-800 hover:bg-slate-700";
    button.textContent = category.charAt(0).toUpperCase() + category.slice(1);
    button.dataset.category = category;
    button.addEventListener("click", () => renderThumbnails(category));
    wrapper.appendChild(button);
    categoryList.appendChild(wrapper);
  });
}

function renderThumbnails(category) {
  thumbnailPane.innerHTML = "";
  (partsByCategory[category] || []).forEach((path) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "thumb-item";

    const image = document.createElement("img");
    image.src = `/static/${path}`;
    image.alt = path;

    item.appendChild(image);
    item.addEventListener("click", () => addLayer(image.src));
    thumbnailPane.appendChild(item);
  });
}

function addLayer(src) {
  const image = new Image();
  image.onload = () => {
    layers.push({
      image,
      x: (canvas.width - image.width) / 2,
      y: (canvas.height - image.height) / 2,
      width: image.width,
      height: image.height,
    });
    activeLayerIndex = layers.length - 1;
    drawCanvas();
  };
  image.src = src;
}

function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  layers.forEach((layer) => {
    ctx.drawImage(layer.image, layer.x, layer.y, layer.width, layer.height);
  });
}

function hitTest(x, y) {
  for (let i = layers.length - 1; i >= 0; i -= 1) {
    const layer = layers[i];
    if (
      x >= layer.x &&
      x <= layer.x + layer.width &&
      y >= layer.y &&
      y <= layer.y + layer.height
    ) {
      return i;
    }
  }
  return -1;
}

canvas.addEventListener("mousedown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  const targetIndex = hitTest(x, y);
  if (targetIndex >= 0) {
    const [selected] = layers.splice(targetIndex, 1);
    layers.push(selected);
    activeLayerIndex = layers.length - 1;
    dragOffset.x = x - selected.x;
    dragOffset.y = y - selected.y;
    drawCanvas();
  }
});

canvas.addEventListener("mousemove", (event) => {
  if (activeLayerIndex < 0 || event.buttons !== 1) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  const activeLayer = layers[activeLayerIndex];
  activeLayer.x = x - dragOffset.x;
  activeLayer.y = y - dragOffset.y;
  drawCanvas();
});

canvas.addEventListener("mouseup", () => {
  activeLayerIndex = -1;
});

canvas.addEventListener("mouseleave", () => {
  activeLayerIndex = -1;
});

resetButton.addEventListener("click", () => {
  layers = [];
  activeLayerIndex = -1;
  drawCanvas();
});

saveButton.addEventListener("click", async () => {
  const payload = { image: canvas.toDataURL("image/png") };

  const response = await fetch("/save_sketch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    alert(data.error || "Unable to save sketch.");
    return;
  }

  alert(`Sketch saved: ${data.filename}`);
});

loadParts();
