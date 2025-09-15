// empty - inline JS used in template for simplicity
// static/js/app.js
async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/upload/", { method: "POST", body: fd });
  return r.json();
}

async function askQuestion(question) {
  const r = await fetch("/api/chat/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  return r.json();
}
