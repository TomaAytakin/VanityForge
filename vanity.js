document.addEventListener('DOMContentLoaded', () => {
    initLiveActivityCounter();
    checkAdmin();
});

function initLiveActivityCounter() {
    const counterElement = document.getElementById('identity-counter');
    const gearIcon = document.getElementById('activity-gear');

    if (!counterElement || !gearIcon) return;

    const START_DATE = new Date("2024-12-01T00:00:00Z");
    const BASE_COUNT = 4242;
    const DAILY_RATE = 45;

    function getDeterministicCount() {
        const now = new Date();
        const diffTime = now.getTime() - START_DATE.getTime();
        // Convert milliseconds to days
        const daysElapsed = diffTime / (1000 * 3600 * 24);

        // Micro-jitter: based on current UTC hour
        const currentHour = now.getUTCHours();
        const jitter = (currentHour * 7) % 13;

        // Calculate total
        const count = Math.floor(BASE_COUNT + (daysElapsed * DAILY_RATE) + jitter);
        return count;
    }

    let currentCount = getDeterministicCount();

    function updateDisplay(count) {
        // Format with commas
        counterElement.innerText = count.toLocaleString();
    }

    // Initial display
    updateDisplay(currentCount);

    // Live Pulse Logic
    function scheduleNextPulse() {
        // Random delay between 10s (10000ms) and 30s (30000ms)
        const delay = Math.floor(Math.random() * (30000 - 10000 + 1) + 10000);
        setTimeout(() => {
            pulse();
        }, delay);
    }

    function pulse() {
        currentCount++;
        updateDisplay(currentCount);

        // Speed up gear
        gearIcon.classList.add('gear-fast');
        setTimeout(() => {
            gearIcon.classList.remove('gear-fast');
        }, 2000); // Spin fast for 2 seconds

        scheduleNextPulse();
    }

    // Start the loop
    scheduleNextPulse();
}

// --- GOD MODE / ADMIN DASHBOARD LOGIC ---
function checkAdmin() {
    // 1. Check for the weak cookie flag
    const isAdmin = document.cookie.split(';').some((item) => item.trim().startsWith('is_admin_flag=1'));

    if (isAdmin) {
        // 2. Inject Navbar Item
        const navContainer = document.querySelector('nav .md\\:block .ml-10');
        if (navContainer) {
            const devBtn = document.createElement('a');
            devBtn.href = "#";
            devBtn.id = "nav-dev";
            devBtn.onclick = (e) => { e.preventDefault(); openAdminDashboard(); };
            devBtn.className = "text-yellow-400 hover:text-white hover:bg-yellow-900/20 px-3 py-2 rounded-md text-sm font-bold transition-colors animate-pulse";
            devBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> DEV';
            navContainer.appendChild(devBtn);
        }
    }
}

async function openAdminDashboard() {
    // Reveal parent container if needed (God Mode Bypass)
    const appInterface = document.getElementById('app-interface');
    if (appInterface.classList.contains('hidden')) {
        document.getElementById('auth-section').classList.add('hidden');
        appInterface.classList.remove('hidden');
    }

    // Hide others
    const sections = ['forge-section', 'referral-dashboard', 'job-queue-section'];
    sections.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('hidden');
    });

    const dash = document.getElementById('admin-dashboard');
    if (dash) dash.classList.remove('hidden');

    // Fetch Initial Stats
    showAdminTab('financials');
}

window.showAdminTab = async function(tabName) {
    const tabs = ['financials', 'security', 'referrals-admin'];
    tabs.forEach(t => {
        const el = document.getElementById(`admin-${t}`);
        if(el) {
            if (t === tabName) el.classList.remove('hidden');
            else el.classList.add('hidden');
        }
    });

    if (tabName === 'financials') {
        try {
            const res = await fetch('/api/admin/stats');
            const data = await res.json();
            if (data.error) return alert("Auth Failed");
            document.getElementById('admin-inc-1w').innerText = data.income_1w + " SOL";
            document.getElementById('admin-inc-1m').innerText = data.income_1m + " SOL";
            document.getElementById('admin-users').innerText = data.total_users;
        } catch(e) { console.error(e); }
    } else if (tabName === 'security') {
        try {
            const res = await fetch('/api/admin/security');
            const data = await res.json();
            const logDiv = document.getElementById('admin-sec-logs');
            logDiv.innerHTML = data.map(l => `<div>[${l.time}] <strong>${l.ip}</strong>: ${l.reason}</div>`).join('');
        } catch(e) { console.error(e); }
    } else if (tabName === 'referrals-admin') {
         try {
            const res = await fetch('/api/admin/referrals');
            const data = await res.json();
            const listDiv = document.getElementById('admin-ref-list');
            listDiv.innerHTML = data.map(r => `
                <div class="flex justify-between bg-gray-800 p-2 rounded text-xs">
                    <span class="font-mono text-purple-400">${r.code}</span>
                    <span class="text-green-400">${r.earnings} SOL</span>
                </div>
            `).join('');
        } catch(e) { console.error(e); }
    }
};
