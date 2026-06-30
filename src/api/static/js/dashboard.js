/* ==================== GLOBAL STATE ==================== */
let currentPage = 'dashboard';
let chartsCache = {};
let dataRefreshInterval = null;
let monitorPollInterval = null;
let map = null;
let mapMarkers = [];
let latestPredictionAlert = null;
let focusOnLatest = false;

const API_BASE = '/api';

/* ==================== AUTO-MONITORING ==================== */
async function toggleMonitoring() {
    const btn = document.getElementById('monitorToggleBtn');
    const badge = document.getElementById('monitorBadge');

    try {
        const statusRes = await fetch(`${API_BASE}/monitoring/status`);
        const status = await statusRes.json();

        if (status.running) {
            // Stop
            const res = await fetch(`${API_BASE}/monitoring/stop`, { method: 'POST' });
            const result = await res.json();
            btn.textContent = 'Start Monitoring';
            btn.style.background = '#22c55e';
            badge.textContent = 'STOPPED';
            badge.style.background = '#ef4444';
            if (monitorPollInterval) clearInterval(monitorPollInterval);
            monitorPollInterval = null;
            toast('Automatic monitoring stopped', 'info');
        } else {
            // Start
            const interval = parseInt(document.getElementById('monitorInterval').value) || 120;
            const res = await fetch(`${API_BASE}/monitoring/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ interval })
            });
            const result = await res.json();
            btn.textContent = 'Stop Monitoring';
            btn.style.background = '#ef4444';
            badge.textContent = 'RUNNING';
            badge.style.background = '#22c55e';
            document.getElementById('monitorStats').style.display = 'block';
            toast(`Automatic monitoring started — scanning every ${interval}s`, 'success');

            // Start polling for status updates
            pollMonitorStatus();
            monitorPollInterval = setInterval(pollMonitorStatus, 5000);
        }
    } catch (error) {
        console.error('Monitor toggle error:', error);
        toast('Error toggling monitoring', 'error');
    }
}

async function pollMonitorStatus() {
    try {
        const res = await fetch(`${API_BASE}/monitoring/status`);
        const status = await res.json();

        const badge = document.getElementById('monitorBadge');
        const btn = document.getElementById('monitorToggleBtn');

        if (status.running) {
            badge.textContent = 'RUNNING';
            badge.style.background = '#22c55e';
            btn.textContent = 'Stop Monitoring';
            btn.style.background = '#ef4444';
            document.getElementById('monitorStats').style.display = 'block';
        } else {
            badge.textContent = 'STOPPED';
            badge.style.background = '#ef4444';
            btn.textContent = 'Start Monitoring';
            btn.style.background = '#22c55e';
        }

        document.getElementById('monScans').textContent = status.total_scans || 0;
        document.getElementById('monAlerts').textContent = status.alerts_generated || 0;
        document.getElementById('monNotifs').textContent = status.notifications_sent || 0;
        document.getElementById('monInterval').textContent = (status.interval_seconds || 120) + 's';

        // Update scan log
        const scans = status.recent_scans || [];
        if (scans.length > 0) {
            const logDiv = document.getElementById('monitorLog');
            let html = '';
            for (const scan of scans.reverse()) {
                const time = new Date(scan.timestamp).toLocaleTimeString();
                if (scan.deforestation_detected) {
                    html += `<div style="color:#ef4444; margin-bottom:4px;">
                        [${time}] ALERT #${scan.scan_number}: <strong>${scan.cause}</strong> detected in ${scan.region}
                        — ${scan.area_hectares?.toFixed(1) || '?'} ha, ${scan.severity}
                        ${scan.notifications ? '→ Sent via: ' + scan.notifications.join(', ') : ''}
                    </div>`;
                } else {
                    html += `<div style="margin-bottom:4px;">
                        [${time}] Scan #${scan.scan_number}: No deforestation — ${scan.region} (image #${scan.image_index})
                    </div>`;
                }
            }
            logDiv.innerHTML = html;

            // Also refresh dashboard data when new scans come in
            if (currentPage === 'dashboard') loadDashboardData();
        }
    } catch (error) {
        console.error('Monitor status poll error:', error);
    }
}

/* ==================== PAGE SWITCHING ==================== */
function switchPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    document.getElementById(`page-${page}`).classList.add('active');
    document.querySelector(`[data-page="${page}"]`).classList.add('active');

    const titles = {
        dashboard: 'Dashboard',
        alerts: 'Alerts',
        map: 'Map View',
        officers: 'Officers',
        notifications: 'Notifications',
        predictions: 'Predictions'
    };
    document.getElementById('pageTitle').textContent = titles[page] || 'Dashboard';
    currentPage = page;

    if (page === 'alerts') loadAlerts();
    else if (page === 'map') initializeMap();
    else if (page === 'officers') loadOfficers();
    else if (page === 'notifications') loadNotificationStatus();
    else if (page === 'dashboard') loadDashboardData();

    closeSidebar();
}

function closeSidebar() {
    if (window.innerWidth <= 768) {
        document.getElementById('sidebar').classList.remove('active');
    }
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('active');
}

/* ==================== DASHBOARD DATA ==================== */
async function loadDashboardData() {
    try {
        // Load statistics
        const statsRes = await fetch(`${API_BASE}/alerts/statistics`);
        const stats = await statsRes.json();

        // Parse nested response format
        const total = stats.total_alerts || 0;
        const byStatus = stats.by_status || {};
        const byCause = stats.by_cause || {};
        const bySeverity = stats.by_severity || {};
        const avgConf = stats.average_confidence || 0;

        const active = (byStatus.pending || 0) + (byStatus.sent || 0) + (byStatus.acknowledged || 0) + (byStatus.investigating || 0);

        // Update stat cards
        document.getElementById('totalAlerts').textContent = total;
        document.getElementById('activeAlerts').textContent = active;
        document.getElementById('resolvedAlerts').textContent = byStatus.resolved || 0;
        document.getElementById('avgConfidence').textContent = total > 0 ? (avgConf * 100).toFixed(1) + '%' : '0%';

        // Load officers count
        try {
            const officersRes = await fetch(`${API_BASE}/officers`);
            const officersData = await officersRes.json();
            document.getElementById('totalOfficers').textContent = officersData.count || 0;
        } catch (e) {
            document.getElementById('totalOfficers').textContent = 0;
        }

        // Update charts
        updateCharts(byStatus, byCause, bySeverity);

        // Load recent alerts
        const alertsRes = await fetch(`${API_BASE}/alerts`);
        const alertsData = await alertsRes.json();
        const alerts = alertsData.alerts || [];
        updateRecentAlerts(alerts.slice(0, 5));

        // Update badge
        document.getElementById('alertBadge').textContent = total;

    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

function updateCharts(byStatus, byCause, bySeverity) {
    // Deforestation by Cause Chart
    const causeCtx = document.getElementById('causeChart');
    if (chartsCache.causeChart) chartsCache.causeChart.destroy();

    chartsCache.causeChart = new Chart(causeCtx, {
        type: 'doughnut',
        data: {
            labels: ['Logging', 'Mining', 'Agriculture', 'Fire', 'Infrastructure'],
            datasets: [{
                data: [
                    byCause['Logging'] || 0,
                    byCause['Mining'] || 0,
                    byCause['Agriculture'] || 0,
                    byCause['Fire'] || 0,
                    byCause['Infrastructure'] || 0
                ],
                backgroundColor: ['#8b5cf6', '#f59e0b', '#10b981', '#ef4444', '#06b6d4'],
                borderColor: '#1e293b',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#cbd5e1', font: { size: 12, weight: 500 } }
                }
            }
        }
    });

    // Alert Severity Chart
    const severityCtx = document.getElementById('severityChart');
    if (chartsCache.severityChart) chartsCache.severityChart.destroy();

    chartsCache.severityChart = new Chart(severityCtx, {
        type: 'bar',
        data: {
            labels: ['Low', 'Medium', 'High', 'Critical'],
            datasets: [{
                label: 'Count',
                data: [
                    bySeverity['low'] || 0,
                    bySeverity['medium'] || 0,
                    bySeverity['high'] || 0,
                    bySeverity['critical'] || 0
                ],
                backgroundColor: ['#facc15', '#f97316', '#ef4444', '#7f1d1d'],
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#cbd5e1', font: { size: 11 }, stepSize: 1 },
                    grid: { color: '#334155' },
                    beginAtZero: true
                },
                y: { ticks: { color: '#cbd5e1', font: { size: 11 } } }
            }
        }
    });

    // Alert Status Chart
    const statusCtx = document.getElementById('statusChart');
    if (chartsCache.statusChart) chartsCache.statusChart.destroy();

    chartsCache.statusChart = new Chart(statusCtx, {
        type: 'line',
        data: {
            labels: ['Pending', 'Sent', 'Acknowledged', 'Investigating', 'Resolved'],
            datasets: [{
                label: 'Count',
                data: [
                    byStatus['pending'] || 0,
                    byStatus['sent'] || 0,
                    byStatus['acknowledged'] || 0,
                    byStatus['investigating'] || 0,
                    byStatus['resolved'] || 0
                ],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#10b981',
                pointBorderColor: '#1e293b'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#cbd5e1', font: { size: 12 } } } },
            scales: {
                x: {
                    ticks: { color: '#cbd5e1', font: { size: 11 } },
                    grid: { color: '#334155' }
                },
                y: {
                    ticks: { color: '#cbd5e1', font: { size: 11 }, stepSize: 1 },
                    grid: { color: '#334155' },
                    beginAtZero: true
                }
            }
        }
    });
}

function updateRecentAlerts(alerts) {
    const tbody = document.getElementById('recentAlertsBody');

    if (!alerts || alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-row">No alerts yet. Use "New Prediction" to generate alerts.</td></tr>';
        return;
    }

    tbody.innerHTML = alerts.map(alert => `
        <tr>
            <td>${(alert.alert_id || '').substring(0, 8)}...</td>
            <td>${alert.cause || 'N/A'}</td>
            <td><span class="severity-badge severity-${alert.severity}">${alert.severity}</span></td>
            <td>${alert.affected_area_hectares ? alert.affected_area_hectares.toFixed(2) : '0'}</td>
            <td>${alert.confidence ? (alert.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
            <td>${alert.region || 'N/A'}</td>
            <td><span class="badge badge-${alert.status}">${alert.status}</span></td>
            <td>${alert.timestamp ? new Date(alert.timestamp).toLocaleDateString() : 'N/A'}</td>
        </tr>
    `).join('');
}

/* ==================== ALERTS PAGE ==================== */
async function loadAlerts() {
    try {
        const statusFilter = document.getElementById('alertStatusFilter')?.value || '';
        let url = `${API_BASE}/alerts`;
        if (statusFilter) url += `?status=${statusFilter}`;

        const response = await fetch(url);
        const data = await response.json();
        const alerts = data.alerts || [];

        const tbody = document.getElementById('allAlertsBody');
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="empty-row">No alerts. Run a prediction to generate alerts.</td></tr>';
            return;
        }

        tbody.innerHTML = alerts.map(alert => `
            <tr>
                <td>${(alert.alert_id || '').substring(0, 8)}...</td>
                <td>${alert.cause || 'N/A'}</td>
                <td><span class="severity-badge severity-${alert.severity}">${alert.severity}</span></td>
                <td>${alert.affected_area_hectares ? alert.affected_area_hectares.toFixed(2) : '0'}</td>
                <td>${alert.confidence ? (alert.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                <td>${(alert.latitude || 0).toFixed(2)}, ${(alert.longitude || 0).toFixed(2)}</td>
                <td>${alert.region || 'N/A'}</td>
                <td>${alert.assigned_officer_name || 'Unassigned'}</td>
                <td><span class="badge badge-${alert.status}">${alert.status}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline" onclick="updateAlertStatus('${alert.alert_id}', 'acknowledged')">
                        Ack
                    </button>
                </td>
            </tr>
        `).join('');

        document.getElementById('alertBadge').textContent = alerts.length;

    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

async function updateAlertStatus(alertId, newStatus) {
    try {
        const response = await fetch(`${API_BASE}/alerts/${alertId}/status`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus })
        });

        if (response.ok) {
            toast(`Alert status updated to ${newStatus}`, 'success');
            loadAlerts();
            loadDashboardData();
        }
    } catch (error) {
        console.error('Error updating alert:', error);
        toast('Error updating alert', 'error');
    }
}

/* ==================== MAP PAGE ==================== */
async function initializeMap() {
    if (!map) {
        map = L.map('alertMap').setView([20, 78], 5);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        }).addTo(map);
    }

    // Fix tiles not loading when map was initialized while hidden
    setTimeout(() => { map.invalidateSize(); }, 200);

    try {
        const response = await fetch(`${API_BASE}/alerts`);
        const data = await response.json();
        const alerts = data.alerts || [];

        // Clear existing markers
        mapMarkers.forEach(marker => map.removeLayer(marker));
        mapMarkers = [];

        let latestMarker = null;

        alerts.forEach(alert => {
            const isLatest = latestPredictionAlert && alert.alert_id === latestPredictionAlert.alert_id;

            let color = '#facc15';
            if (alert.severity === 'medium') color = '#f97316';
            else if (alert.severity === 'high') color = '#ef4444';
            else if (alert.severity === 'critical') color = '#7f1d1d';

            const lat = alert.latitude || 0;
            const lon = alert.longitude || 0;
            let marker;

            if (isLatest) {
                // Pulsing animated icon for the latest prediction
                const pulseIcon = L.divIcon({
                    className: '',
                    html: `<div class="map-pulse-container">
                               <div class="map-pulse-ring" style="border-color:${color};"></div>
                               <div class="map-pulse-dot" style="background:${color};"></div>
                           </div>`,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15],
                    popupAnchor: [0, -18]
                });
                marker = L.marker([lat, lon], { icon: pulseIcon }).addTo(map);
            } else {
                marker = L.circleMarker([lat, lon], {
                    radius: 8,
                    fillColor: color,
                    color: '#fff',
                    weight: 1.5,
                    opacity: 0.9,
                    fillOpacity: 0.8
                }).addTo(map);
            }

            // Tooltip: permanent region name label for latest, hover-only for others
            marker.bindTooltip(alert.region || 'Unknown Region', {
                permanent: isLatest,
                direction: 'top',
                className: isLatest ? 'map-label-latest' : 'map-label',
                offset: isLatest ? [0, -20] : [0, -8]
            });

            // Rich popup with full details
            const ts = alert.timestamp ? new Date(alert.timestamp).toLocaleString() : '';
            const officerRow = alert.assigned_officer_name
                ? `<tr><td style="color:#666;padding:3px 0;">Officer</td><td style="font-weight:600;text-align:right;">${alert.assigned_officer_name}</td></tr>`
                : '';
            const tsRow = ts ? `<tr><td colspan="2" style="color:#999;font-size:11px;padding-top:6px;border-top:1px solid #eee;margin-top:4px;">${ts}</td></tr>` : '';

            marker.bindPopup(`
                <div style="min-width:230px;font-family:system-ui,sans-serif;">
                    <div style="background:${color};color:#fff;padding:8px 12px;margin:-8px -12px 10px;border-radius:4px 4px 0 0;">
                        <div style="font-size:14px;font-weight:700;">📍 ${alert.region || 'Unknown Region'}</div>
                        ${isLatest ? '<div style="font-size:11px;opacity:0.9;margin-top:2px;">✨ Latest AI Prediction</div>' : ''}
                    </div>
                    <table style="width:100%;font-size:12px;color:#333;border-collapse:collapse;">
                        <tr><td style="color:#666;padding:3px 0;">Cause</td><td style="font-weight:600;text-align:right;">${alert.cause}</td></tr>
                        <tr><td style="color:#666;padding:3px 0;">Severity</td><td style="font-weight:700;color:${color};text-align:right;">${(alert.severity || '').toUpperCase()}</td></tr>
                        <tr><td style="color:#666;padding:3px 0;">Area</td><td style="font-weight:600;text-align:right;">${alert.affected_area_hectares.toFixed(2)} ha</td></tr>
                        <tr><td style="color:#666;padding:3px 0;">Confidence</td><td style="font-weight:600;text-align:right;">${(alert.confidence * 100).toFixed(1)}%</td></tr>
                        <tr><td style="color:#666;padding:3px 0;">GPS</td><td style="font-weight:600;text-align:right;">${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</td></tr>
                        <tr><td style="color:#666;padding:3px 0;">Status</td><td style="font-weight:600;text-align:right;">${alert.status || 'N/A'}</td></tr>
                        ${officerRow}
                        ${tsRow}
                    </table>
                </div>
            `, { maxWidth: 290 });

            mapMarkers.push(marker);
            if (isLatest) latestMarker = marker;
        });

        // If "View on Map" was clicked from a prediction result, zoom + open popup
        if (focusOnLatest && latestMarker && latestPredictionAlert) {
            focusOnLatest = false;
            map.setView([latestPredictionAlert.latitude, latestPredictionAlert.longitude], 10);
            setTimeout(() => latestMarker.openPopup(), 400);
        } else if (mapMarkers.length > 0) {
            const group = L.featureGroup(mapMarkers);
            map.fitBounds(group.getBounds().pad(0.3));
        }

    } catch (error) {
        console.error('Error loading map data:', error);
    }
}

function viewLatestOnMap() {
    focusOnLatest = true;
    switchPage('map');
}

/* ==================== OFFICERS PAGE ==================== */
async function loadOfficers() {
    try {
        const response = await fetch(`${API_BASE}/officers`);
        const data = await response.json();
        const officers = data.officers || [];

        const grid = document.getElementById('officerGrid');
        if (officers.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    No officers yet.<br><br>
                    <button class="btn btn-primary" onclick="setupDemoOfficers()">Setup Demo Officers</button>
                    &nbsp;
                    <button class="btn btn-outline" onclick="showAddOfficerModal()">+ Add Officer</button>
                </div>`;
            return;
        }

        grid.innerHTML = officers.map(officer => `
            <div class="officer-card">
                <div class="officer-avatar">${(officer.name || 'U').charAt(0)}</div>
                <div class="officer-name">${officer.name}</div>
                <div class="officer-region">${officer.region || 'No region'}</div>
                <div class="officer-info">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="14" height="14">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
                    </svg>
                    ${officer.email || 'No email'}
                </div>
                ${officer.phone ? `<div class="officer-info">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="14" height="14">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/>
                    </svg>
                    ${officer.phone}
                </div>` : ''}
                <div class="officer-alerts">
                    <div class="count">Active</div>
                    <div>${officer.region || 'All regions'}</div>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading officers:', error);
    }
}

function showAddOfficerModal() {
    showModal('Add Officer', `
        <form onsubmit="createOfficer(event)">
            <div class="form-group" style="margin-bottom:12px;">
                <label>Name</label>
                <input type="text" id="officerName" required style="width:100%;">
            </div>
            <div class="form-group" style="margin-bottom:12px;">
                <label>Phone</label>
                <input type="text" id="officerPhone" style="width:100%;">
            </div>
            <div class="form-group" style="margin-bottom:12px;">
                <label>Email</label>
                <input type="email" id="officerEmail" required style="width:100%;">
            </div>
            <div class="form-group" style="margin-bottom:12px;">
                <label>Region</label>
                <input type="text" id="officerRegion" style="width:100%;">
            </div>
            <div class="form-group" style="margin-bottom:12px;">
                <label>Telegram Chat ID</label>
                <input type="text" id="officerTelegram" style="width:100%;">
            </div>
            <button type="submit" class="btn btn-primary btn-lg">Create Officer</button>
        </form>
    `);
}

async function createOfficer(e) {
    e.preventDefault();

    const payload = {
        name: document.getElementById('officerName').value,
        phone: document.getElementById('officerPhone').value,
        email: document.getElementById('officerEmail').value,
        region: document.getElementById('officerRegion').value,
        telegram_chat_id: document.getElementById('officerTelegram').value
    };

    try {
        const response = await fetch(`${API_BASE}/officers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            toast('Officer created successfully', 'success');
            closeModal();
            loadOfficers();
        } else {
            const err = await response.json();
            toast(err.error || 'Error creating officer', 'error');
        }
    } catch (error) {
        console.error('Error creating officer:', error);
        toast('Error creating officer', 'error');
    }
}

async function setupDemoOfficers() {
    try {
        const response = await fetch(`${API_BASE}/officers/setup-demo`, { method: 'POST' });
        if (response.ok) {
            const data = await response.json();
            toast(data.message || 'Demo officers created!', 'success');
            loadOfficers();
            loadDashboardData();
        }
    } catch (error) {
        console.error('Error setting up demo officers:', error);
        toast('Error setting up demo officers', 'error');
    }
}

/* ==================== NOTIFICATIONS PAGE ==================== */
async function loadNotificationStatus() {
    try {
        const response = await fetch(`${API_BASE}/notifications/status`);
        const data = await response.json();
        const status = data.status || {};

        // FCM
        const fcmMode = status.fcm ? status.fcm.mode : 'demo';
        document.getElementById('fcmStatus').textContent = fcmMode === 'live' ? 'Active - Live Mode' : 'Demo Mode (disabled)';
        document.getElementById('fcmBadge').textContent = fcmMode === 'live' ? 'Live' : 'Demo';

        // Telegram
        const telegramMode = status.telegram ? status.telegram.mode : 'demo';
        document.getElementById('telegramStatus').textContent = telegramMode === 'live' ? 'Active - Bot Connected' : 'Demo Mode (add bot token to .env)';
        document.getElementById('telegramBadge').textContent = telegramMode === 'live' ? 'Live' : 'Demo';

        // Email
        const emailMode = status.email ? status.email.mode : 'demo';
        document.getElementById('emailStatus').textContent = emailMode === 'live' ? 'Active - Gmail SMTP' : 'Demo Mode (add Gmail app password to .env)';
        document.getElementById('emailBadge').textContent = emailMode === 'live' ? 'Live' : 'Demo';

    } catch (error) {
        console.error('Error loading notification status:', error);
    }
}

async function testAllTiers() {
    try {
        toast('Sending test notifications...', 'info');
        const response = await fetch(`${API_BASE}/notifications/test`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            toast('Test notifications sent (check logs for demo mode)', 'success');
        } else {
            toast(data.error || 'Error testing notifications', 'warning');
        }
    } catch (error) {
        toast('Error testing notifications', 'error');
    }
}

/* ==================== PREDICTIONS PAGE ==================== */
async function runPrediction() {
    const btn = document.getElementById('runPredBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    try {
        const payload = {
            cause: document.getElementById('predCause').value,
            latitude: parseFloat(document.getElementById('predLat').value),
            longitude: parseFloat(document.getElementById('predLon').value),
            region: document.getElementById('predRegion').value,
            area_fraction: parseFloat(document.getElementById('predArea').value)
        };

        const response = await fetch(`${API_BASE}/predictions/demo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.deforestation_detected && result.alert) {
            const alert = result.alert;
            latestPredictionAlert = alert;
            const content = `
                <div style="background:#1e293b; padding:16px; border-radius:8px;">
                    <div style="margin-bottom:16px; text-align:center;">
                        <span style="color:#ef4444; font-size:18px; font-weight:700;">Deforestation Detected!</span>
                    </div>
                    <div style="background:#334155; padding:12px; border-radius:6px; margin-bottom:12px;">
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:13px;">
                            <div>
                                <span style="color:#94a3b8;">Cause</span><br>
                                <strong style="font-size:16px; color:#f59e0b;">${alert.cause}</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Severity</span><br>
                                <strong style="font-size:16px;" class="severity-badge severity-${alert.severity}">${alert.severity}</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Affected Area</span><br>
                                <strong style="font-size:16px;">${alert.affected_area_hectares.toFixed(2)} ha</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Confidence</span><br>
                                <strong style="font-size:16px;">${(alert.confidence * 100).toFixed(1)}%</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Location</span><br>
                                <strong>${(alert.latitude || 0).toFixed(2)}, ${(alert.longitude || 0).toFixed(2)}</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Region</span><br>
                                <strong>${alert.region}</strong>
                            </div>
                        </div>
                    </div>
                    <div style="background:#334155; padding:10px; border-radius:6px; margin-bottom:12px; font-size:12px; color:#94a3b8;">
                        Alert ID: ${alert.alert_id}<br>
                        Status: <span class="badge badge-${alert.status}">${alert.status}</span>
                    </div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        <button class="btn btn-primary" onclick="switchPage('alerts')">View in Alerts</button>
                        <button class="btn btn-outline" onclick="viewLatestOnMap()">📍 View on Map</button>
                    </div>
                </div>
            `;
            document.getElementById('predResultContent').innerHTML = content;
            toast('Prediction complete - Alert generated!', 'success');
        } else {
            document.getElementById('predResultContent').innerHTML = `
                <div style="text-align:center; padding:32px; color:#94a3b8;">
                    <p style="font-size:16px; margin-bottom:8px;">No significant deforestation detected</p>
                    <p style="font-size:12px;">${result.message || 'Try increasing the area fraction'}</p>
                </div>
            `;
            toast('Prediction complete - No deforestation detected', 'info');
        }

    } catch (error) {
        console.error('Error running prediction:', error);
        toast('Error running prediction', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Prediction';
    }
}

async function generateDemoPrediction() {
    try {
        const response = await fetch(`${API_BASE}/predictions/demo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        if (response.ok) {
            const result = await response.json();
            if (result.deforestation_detected) {
                toast('Demo prediction created - Alert generated!', 'success');
                loadDashboardData();
            } else {
                toast('Demo prediction: No deforestation detected', 'info');
            }
        }
    } catch (error) {
        toast('Error creating demo prediction', 'error');
    }
}

async function runSatellitePrediction() {
    const btn = document.getElementById('runSatelliteBtn');
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = 'Analyzing Satellite Image...';

    try {
        const response = await fetch(`${API_BASE}/predictions/satellite`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const result = await response.json();

        if (result.error) {
            document.getElementById('predResultContent').innerHTML = `
                <div style="text-align:center; padding:32px; color:#ef4444;">
                    <p style="font-size:16px;">${result.error}</p>
                    <p style="font-size:12px; color:#94a3b8;">${result.message || ''}</p>
                </div>`;
            toast(result.error, 'error');
            return;
        }

        if (result.deforestation_detected && result.alert) {
            const alert = result.alert;
            latestPredictionAlert = alert;
            const summary = result.model_summary || {};
            const notifs = result.notifications_sent || [];
            const content = `
                <div style="background:#1e293b; padding:16px; border-radius:8px;">
                    <div style="margin-bottom:12px; text-align:center;">
                        <span style="color:#ef4444; font-size:18px; font-weight:700;">U-Net Model: Deforestation Detected!</span>
                        <div style="color:#94a3b8; font-size:11px; margin-top:4px;">
                            Source: Real satellite image #${result.image_index} &bull; 11-band Sentinel-1/2 data &bull;
                            Ground truth accuracy: <strong style="color:#22c55e;">${result.ground_truth_accuracy}%</strong>
                        </div>
                    </div>
                    <div style="background:#0f172a; padding:10px; border-radius:6px; margin-bottom:12px; font-size:12px; color:#60a5fa; border:1px solid #1e40af;">
                        <strong>Satellite Bands Used:</strong> ${(result.satellite_bands || []).join(', ')}
                    </div>
                    <div style="background:#334155; padding:12px; border-radius:6px; margin-bottom:12px;">
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:13px;">
                            <div>
                                <span style="color:#94a3b8;">Cause (AI Detected)</span><br>
                                <strong style="font-size:16px; color:#f59e0b;">${alert.cause}</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Severity</span><br>
                                <strong style="font-size:16px;" class="severity-badge severity-${alert.severity}">${alert.severity}</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Affected Area</span><br>
                                <strong style="font-size:16px;">${alert.affected_area_hectares.toFixed(2)} ha</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">AI Confidence</span><br>
                                <strong style="font-size:16px;">${(alert.confidence * 100).toFixed(1)}%</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">GPS Location</span><br>
                                <strong>${(alert.latitude || 0).toFixed(4)}, ${(alert.longitude || 0).toFixed(4)}</strong>
                            </div>
                            <div>
                                <span style="color:#94a3b8;">Forest Region</span><br>
                                <strong>${alert.region}</strong>
                            </div>
                        </div>
                    </div>
                    <div style="background:#334155; padding:10px; border-radius:6px; margin-bottom:12px;">
                        <div style="font-size:12px; color:#94a3b8; margin-bottom:6px;"><strong>Model Analysis Summary</strong></div>
                        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; font-size:12px;">
                            <div>Total: <strong>${(summary.total_area_hectares || 0).toFixed(1)} ha</strong></div>
                            <div>Forest: <strong style="color:#22c55e;">${(summary.forest_area_hectares || 0).toFixed(1)} ha</strong></div>
                            <div>Deforested: <strong style="color:#ef4444;">${(summary.deforestation_area_hectares || 0).toFixed(1)} ha</strong></div>
                        </div>
                    </div>
                    ${notifs.length > 0 ? `
                    <div style="background:#064e3b; padding:8px 12px; border-radius:6px; margin-bottom:12px; font-size:12px; color:#34d399;">
                        Notifications sent via: <strong>${notifs.join(', ')}</strong>
                    </div>` : ''}
                    <div style="display:flex; gap:8px;">
                        <button class="btn btn-primary" onclick="switchPage('alerts')">View in Alerts</button>
                        <button class="btn btn-outline" onclick="viewLatestOnMap()">📍 View on Map</button>
                        <button class="btn btn-outline" onclick="runSatellitePrediction()">Analyze Another</button>
                    </div>
                </div>
            `;
            document.getElementById('predResultContent').innerHTML = content;
            toast('Real satellite analysis complete - Alert & Notifications sent!', 'success');
        } else {
            document.getElementById('predResultContent').innerHTML = `
                <div style="text-align:center; padding:32px; color:#94a3b8;">
                    <p style="font-size:16px;">Model detected no significant deforestation</p>
                    <p style="font-size:12px;">Image #${result.image_index} - ${result.message || 'Forest appears healthy'}</p>
                    <button class="btn btn-outline" style="margin-top:12px;" onclick="runSatellitePrediction()">Try Another Image</button>
                </div>`;
            toast('Model: No deforestation in this satellite image', 'info');
        }

    } catch (error) {
        console.error('Error running satellite prediction:', error);
        toast('Error running satellite analysis', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Satellite Image (Real AI)';
    }
}

/* ==================== MODAL ==================== */
function showModal(title, content) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').innerHTML = content;
    document.getElementById('modalOverlay').classList.add('active');
}

function closeModal() {
    document.getElementById('modalOverlay').classList.remove('active');
}

/* ==================== TOAST NOTIFICATIONS ==================== */
function toast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    const id = 'toast-' + Date.now();

    const icons = {
        success: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>',
        error: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>',
        info: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
        warning: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>'
    };

    const titles = { success: 'Success', error: 'Error', info: 'Info', warning: 'Warning' };

    const html = `
        <div class="toast ${type}" id="${id}">
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-content">
                <div class="toast-title">${titles[type] || 'Info'}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="document.getElementById('${id}').remove()">&times;</button>
        </div>
    `;

    container.insertAdjacentHTML('beforeend', html);

    if (duration > 0) {
        setTimeout(() => {
            const el = document.getElementById(id);
            if (el) el.remove();
        }, duration);
    }
}

/* ==================== SYSTEM STATUS ==================== */
async function updateSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            document.getElementById('systemStatusDot').className = 'status-dot online';
            document.getElementById('systemStatusText').textContent = 'System Online';
        } else {
            throw new Error('Health check failed');
        }
    } catch (error) {
        document.getElementById('systemStatusDot').className = 'status-dot offline';
        document.getElementById('systemStatusText').textContent = 'System Offline';
    }
}

/* ==================== UTILS ==================== */
function refreshData() {
    if (currentPage === 'dashboard') loadDashboardData();
    else if (currentPage === 'alerts') loadAlerts();
    else if (currentPage === 'officers') loadOfficers();
    else if (currentPage === 'map') initializeMap();
    else if (currentPage === 'notifications') loadNotificationStatus();
    toast('Data refreshed', 'success', 2000);
}

/* ==================== INITIALIZATION ==================== */
document.addEventListener('DOMContentLoaded', () => {
    updateSystemStatus();
    loadDashboardData();
    pollMonitorStatus();
    dataRefreshInterval = setInterval(updateSystemStatus, 30000);

    document.getElementById('modalOverlay').addEventListener('click', function(e) {
        if (e.target === this) closeModal();
    });

    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            document.getElementById('sidebar').classList.remove('active');
        }
    });
});

window.addEventListener('beforeunload', () => {
    clearInterval(dataRefreshInterval);
    Object.values(chartsCache).forEach(chart => {
        if (chart) chart.destroy();
    });
});
