document.addEventListener('DOMContentLoaded', () => {
    initLiveActivityCounter();
    checkAdmin();
});

function initLiveActivityCounter() {
    const counterElement = document.getElementById('identity-counter');
    const gearIcon = document.getElementById('activity-gear');

    if (!counterElement || !gearIcon) return;

    let currentCount = 0; // Default until fetch
    let currentStr = "0";
    counterElement.innerHTML = ''; // Clear text
    let digitElements = [];

    // Create a slot for each char (digit or comma)
    for (let char of currentStr) {
        let span = document.createElement('span');
        if (char === ',') {
             span.innerText = ',';
             span.className = 'odometer-val';
             counterElement.appendChild(span);
        } else {
             // Create wrapper
             let wrapper = document.createElement('span');
             wrapper.className = 'odometer-digit';
             // Create ribbon (01234567890)
             let ribbon = document.createElement('span');
             ribbon.className = 'odometer-ribbon';
             ribbon.innerText = '0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n0';
             // We need vertical stacking. textContent with \n works if we set white-space: pre
             // Or better, use innerHTML with <br> or block display
             ribbon.innerHTML = '0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>0';

             wrapper.appendChild(ribbon);
             counterElement.appendChild(wrapper);

             // Initial position
             let digit = parseInt(char);
             ribbon.style.transform = `translateY(-${digit * 1.2}em)`;
             digitElements.push({ wrapper, ribbon, currentDigit: digit });
        }
    }

    function updateDisplay(newCount) {
        const newStr = newCount.toLocaleString();

        // If length changed, rebuild (simple fallback)
        if (newStr.length !== currentStr.length) {
             counterElement.innerHTML = '';
             digitElements = [];
             for (let char of newStr) {
                if (char === ',') {
                    let span = document.createElement('span');
                    span.innerText = ',';
                    span.className = 'odometer-val';
                    counterElement.appendChild(span);
                } else {
                    let wrapper = document.createElement('span');
                    wrapper.className = 'odometer-digit';
                    let ribbon = document.createElement('span');
                    ribbon.className = 'odometer-ribbon';
                    ribbon.innerHTML = '0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>0';
                    wrapper.appendChild(ribbon);
                    counterElement.appendChild(wrapper);

                    let digit = parseInt(char);
                    ribbon.style.transform = `translateY(-${digit * 1.2}em)`;
                    digitElements.push({ wrapper, ribbon, currentDigit: digit });
                }
             }
             currentStr = newStr;
             return;
        }

        // Animate changes
        let digitIndex = 0;
        for (let i = 0; i < newStr.length; i++) {
            if (newStr[i] === ',') continue;

            let targetDigit = parseInt(newStr[i]);
            let el = digitElements[digitIndex];

            if (el.currentDigit !== targetDigit) {
                // Determine direction. Usually roll UP.
                // If going from 9 to 0, we move to the 11th slot (10=0) then reset?
                // Simplest CSS transition: just move to new Y.
                // If wrapping (e.g. 9->0), we might need logic, but for simple counter incrementing:
                // usually we just set the transform.

                // Let's assume standard movement for now.
                el.ribbon.style.transform = `translateY(-${targetDigit * 1.2}em)`;
                el.currentDigit = targetDigit;
            }
            digitIndex++;
        }
        currentStr = newStr;
    }

    // Live Pulse Logic via Backend Fetch
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (res.ok) {
                const data = await res.json();
                const newCount = data.forged_count;

                if (newCount !== currentCount) {
                    currentCount = newCount;
                    updateDisplay(currentCount);

                    // Speed up gear animation briefly
                    gearIcon.classList.add('gear-fast');
                    setTimeout(() => {
                        gearIcon.classList.remove('gear-fast');
                    }, 2000);
                }
            }
        } catch (e) {
            console.error("Stats fetch failed", e);
        }
    }

    // Initial Fetch
    fetchStats();

    // Poll every 60 seconds
    setInterval(fetchStats, 60000);
}

// --- GOD MODE / ADMIN DASHBOARD LOGIC ---
async function checkAdmin() {
    try {
        const res = await fetch('/api/user-status');
        const data = await res.json();

        if (data.isAdmin) {
            // 2. Inject Navbar Item
            const navContainer = document.querySelector('nav .md\\:block .ml-10');
            if (navContainer && !document.getElementById('nav-dev')) {
                const devBtn = document.createElement('a');
                devBtn.href = "#";
                devBtn.id = "nav-dev";
                devBtn.onclick = (e) => { e.preventDefault(); toggleAdminDashboard(); };
                devBtn.className = "text-yellow-400 hover:text-white hover:bg-yellow-900/20 px-3 py-2 rounded-md text-sm font-bold transition-colors animate-pulse";
                devBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> DEV';
                navContainer.appendChild(devBtn);
            }

            // Bind to "GOD MODE ACTIVE" text if present (for mobile/alternate access)
            setTimeout(() => {
                const spans = document.getElementsByTagName('span');
                for (let span of spans) {
                    if (span.innerText.includes('God Mode Active')) {
                        span.style.cursor = 'pointer';
                        span.onclick = toggleAdminDashboard;
                    }
                }
            }, 2000); // Wait for UI to settle
        }
    } catch (e) {
        console.error("Admin check failed", e);
    }
}

function toggleAdminDashboard() {
    const dash = document.getElementById('admin-dashboard');
    if (dash && !dash.classList.contains('hidden')) {
        // Close it
        dash.classList.add('hidden');
        document.getElementById('app-interface').classList.remove('hidden'); // Ensure app is visible
        document.getElementById('job-queue-section').classList.remove('hidden');
        document.getElementById('forge-section').classList.remove('hidden');
    } else {
        openAdminDashboard();
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
