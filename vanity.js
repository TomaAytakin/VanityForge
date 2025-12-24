document.addEventListener('DOMContentLoaded', () => {
    initLiveActivityCounter();
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
