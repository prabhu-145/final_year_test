const input = document.getElementById("sketch");
const preview = document.getElementById("previewImage");

if (input && preview) {
  input.addEventListener("change", () => {
    const [file] = input.files;
    if (!file) {
      preview.classList.add("hidden");
      preview.removeAttribute("src");
      return;
    }

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove("hidden");
  });
}
