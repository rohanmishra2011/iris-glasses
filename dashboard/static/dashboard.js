const byId = (id) => document.getElementById(id);
const humanStatus = (value) => String(value || "unknown").replaceAll("_", " ");

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    byId("connection").textContent = "Live";
    byId("connection").className = "pill ok";
    byId("camera").textContent = humanStatus(state.camera);
    byId("yolo").textContent = humanStatus(state.yolo);
    byId("room").textContent = state.room || "unknown";
    byId("room-status").textContent = humanStatus(state.room_status);
    byId("cpu-temperature").textContent = Number.isFinite(state.cpu_temperature_c)
      ? `${state.cpu_temperature_c.toFixed(1)} °C` : "--";
    byId("supply-voltage").textContent = Number.isFinite(state.supply_voltage_v)
      ? `${state.supply_voltage_v.toFixed(3)} V` : "--";
    byId("power-status").textContent = humanStatus(state.power_status);
    byId("imu-status").textContent = humanStatus(state.imu_status);
    byId("imu-address").textContent = state.imu_address || "--";
    byId("imu-accel").textContent = formatAxes(state.imu_accel_g, 3);
    byId("imu-gyro").textContent = formatAxes(state.imu_gyro_dps, 2);
    byId("imu-temperature").textContent = Number.isFinite(state.imu_temperature_c)
      ? `${state.imu_temperature_c.toFixed(1)} °C` : "--";
    byId("imu-error").textContent = state.imu_error || "";
    byId("voice-status").textContent = humanStatus(state.voice_status);
    byId("last-command").textContent = state.last_command || "Waiting for a command";
    byId("last-response").textContent = state.last_response || "--";
    byId("updated").textContent = state.last_update
      ? `Updated ${new Date(state.last_update).toLocaleTimeString()}` : "Waiting";
    const detections = state.detections || [];
    byId("count").textContent = `${detections.length} object${detections.length === 1 ? "" : "s"}`;
    byId("detections").innerHTML = detections.length ? detections.map((item) => `
      <tr><td>${escapeHtml(item.label)}</td><td>${Math.round(item.confidence * 100)}%</td>
      <td>${item.box.map((v) => Math.round(v)).join(", ")}</td></tr>`).join("")
      : '<tr><td colspan="3" class="empty">No objects detected</td></tr>';
  } catch (error) {
    byId("connection").textContent = "Disconnected";
    byId("connection").className = "pill warning";
  }
}

function formatAxes(value, digits) {
  if (!value) return "x --   y --   z --";
  return ["x", "y", "z"].map((axis) =>
    `${axis} ${Number(value[axis]).toFixed(digits)}`).join("   ");
}

function escapeHtml(value) {
  const node = document.createElement("span"); node.textContent = value; return node.innerHTML;
}

refresh(); setInterval(refresh, 1000);
