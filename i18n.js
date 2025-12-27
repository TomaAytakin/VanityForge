// i18n.js
const I18n = {
    locale: 'en',
    translations: {},

    init: async function() {
        // Load translations
        try {
            const [en, zh] = await Promise.all([
                fetch('locales/en.json').then(r => r.json()),
                fetch('locales/zh-CN.json').then(r => r.json())
            ]);
            this.translations['en'] = en;
            this.translations['zh-CN'] = zh;
        } catch (e) {
            console.error("Failed to load translations", e);
            return;
        }

        // Detect language
        const saved = localStorage.getItem('vf_locale');
        if (saved) {
            this.locale = saved;
        } else {
            const browserLang = navigator.language || navigator.userLanguage;
            if (browserLang && browserLang.toLowerCase().startsWith('zh')) {
                this.locale = 'zh-CN';
            }
        }

        this.apply();
        this.updateToggleUI();
    },

    setLocale: function(newLocale) {
        if (!this.translations[newLocale]) return;
        this.locale = newLocale;
        localStorage.setItem('vf_locale', newLocale);
        this.apply();
        this.updateToggleUI();
    },

    toggle: function() {
        const next = this.locale === 'en' ? 'zh-CN' : 'en';
        this.setLocale(next);
    },

    t: function(key) {
        const keys = key.split('.');
        let val = this.translations[this.locale];
        for (const k of keys) {
            val = val ? val[k] : null;
        }
        return val || key;
    },

    apply: function() {
        // Update elements with data-i18n
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const text = this.t(key);
            if (text) {
                // If it's a security description, use innerHTML to allow tags (if present)
                if (key.includes('_desc')) {
                    el.innerHTML = text;
                } else {
                    if (el.children.length === 0) {
                        el.innerText = text;
                    } else {
                        let textNode = null;
                        for (let node of el.childNodes) {
                            if (node.nodeType === 3 && node.nodeValue.trim().length > 0) {
                                 textNode = node;
                                 break;
                            }
                        }
                        if (textNode) {
                            textNode.nodeValue = text;
                        } else {
                            console.warn("Element with data-i18n has children but no text node found to replace:", el);
                        }
                    }
                }
            }
        });

        // 1. Placeholder
        const prefixInput = document.getElementById('prefix');
        const suffixInput = document.getElementById('suffix');
        if (prefixInput) {
            if (this.locale === 'zh-CN') {
                prefixInput.placeholder = "8888";
                if(suffixInput) suffixInput.placeholder = "666";
            } else {
                prefixInput.placeholder = "e.g. CAFE";
                if(suffixInput) suffixInput.placeholder = "e.g. 888";
            }
        }

        // 2. Fonts
        if (this.locale === 'zh-CN') {
            document.body.classList.add('font-cn');
            document.body.classList.remove('font-en');
        } else {
            document.body.classList.add('font-en');
            document.body.classList.remove('font-cn');
        }

        // 3. God Mode Badge
        const godModeBadge = document.getElementById('god-mode-badge');
        if (godModeBadge) {
            godModeBadge.innerText = this.t('nav.godmode');
        }
    },

    updateToggleUI: function() {
        const btn = document.getElementById('lang-toggle');
        if (btn) {
            // "Add a clean toggle ... Label it 'EN / 中文'"
            if (this.locale === 'en') {
                btn.innerHTML = '<span class="text-white font-bold">EN</span> / <span class="text-gray-500">中文</span>';
            } else {
                btn.innerHTML = '<span class="text-gray-500">EN</span> / <span class="text-white font-bold">中文</span>';
            }
        }
    }
};

window.I18n = I18n;
document.addEventListener('DOMContentLoaded', () => I18n.init());
