document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeToggle');

    // --- 夜间模式 (Dark Mode) ---
    if (themeToggle) {
        // 1. 检查本地存储
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggle.querySelector('.theme-icon').textContent = '☀️';
        }

        // 2. 点击切换
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            
            // 更新图标
            themeToggle.querySelector('.theme-icon').textContent = isDark ? '☀️' : '🌙';
            
            // 保存设置
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });
    }

    // --- 简单的淡入动画 ---
    const cards = document.querySelectorAll('.paper-card, .dataset-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(card);
    });
});