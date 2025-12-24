document.addEventListener('DOMContentLoaded', () => {
    // --- 1. 元素获取 ---
    const searchInput = document.getElementById('searchInput');
    const yearFilter = document.getElementById('yearFilter');
    const filterBtns = document.querySelectorAll('.filter-btn');
    const themeToggle = document.getElementById('themeToggle');
    const allPapers = document.querySelectorAll('.paper-card');
    const allSections = document.querySelectorAll('.papers-section');

    // --- 2. 搜索与筛选核心逻辑 ---
    function filterPapers() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedYear = yearFilter.value;
        // 获取当前激活的分类按钮的类别 (all, conventional, vlm, hybrid)
        const activeCategoryBtn = document.querySelector('.filter-btn.active');
        const selectedCategory = activeCategoryBtn ? activeCategoryBtn.dataset.category : 'all';

        allPapers.forEach(paper => {
            // 1. 获取卡片信息
            const title = paper.querySelector('h4').textContent.toLowerCase();
            const desc = paper.querySelector('.paper-description').textContent.toLowerCase();
            const tags = paper.textContent.toLowerCase(); // 简单粗暴包含所有文本
            const year = paper.dataset.year;
            
            // 2. 判断是否匹配搜索词
            const matchesSearch = title.includes(searchTerm) || 
                                  desc.includes(searchTerm) || 
                                  tags.includes(searchTerm);
            
            // 3. 判断是否匹配年份
            const matchesYear = selectedYear === 'all' || year === selectedYear;

            // 4. 判断是否匹配分类 (通过父级 Section 的 data-category 判断)
            // 注意：HTML结构中，卡片是在 section 里的，我们通过 section 的显隐来控制大类，
            // 但这里为了搜索体验，我们也可以直接控制卡片。
            // 为了简单，我们主要依赖 Section 的显隐来做分类，这里只做搜索和年份。
            
            if (matchesSearch && matchesYear) {
                paper.style.display = 'block';
                // 加上淡入动画效果
                paper.style.animation = 'fadeIn 0.5s ease';
            } else {
                paper.style.display = 'none';
            }
        });

        // 额外处理：如果一个 Section 下所有卡片都隐藏了，是否隐藏该 Section 标题？
        // 暂时保留标题，避免布局跳动太大，或者可以根据需求隐藏
        updateSectionVisibility(selectedCategory);
    }

    // --- 3. 分类切换逻辑 ---
    function updateSectionVisibility(category) {
        allSections.forEach(section => {
            const sectionCategory = section.dataset.category;
            // 如果选的是 'all'，或者 section 的分类等于当前选的分类，就显示
            if (category === 'all' || sectionCategory === category) {
                section.style.display = 'block';
            } else {
                section.style.display = 'none';
            }
        });
    }

    // --- 4. 事件监听绑定 ---
    
    // 搜索框输入事件
    if (searchInput) {
        searchInput.addEventListener('input', filterPapers);
    }

    // 年份下拉改变事件
    if (yearFilter) {
        yearFilter.addEventListener('change', filterPapers);
    }

    // 分类按钮点击事件
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // 移除所有按钮的 active 类
            filterBtns.forEach(b => b.classList.remove('active'));
            // 给当前点击的按钮加 active
            btn.classList.add('active');
            
            // 执行筛选
            filterPapers();
        });
    });

    // --- 5. 夜间模式切换 ---
    if (themeToggle) {
        // 检查本地存储中的偏好
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggle.querySelector('.theme-icon').textContent = '☀️';
        }

        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            
            // 更新图标
            themeToggle.querySelector('.theme-icon').textContent = isDark ? '☀️' : '🌙';
            
            // 保存偏好
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });
    }

    // --- 6. 数字滚动动画 (Hero Section) ---
    const stats = document.querySelectorAll('.stat-number');
    stats.forEach(stat => {
        const target = +stat.dataset.target;
        const duration = 2000; // 2秒
        const increment = target / (duration / 16); // 60fps
        
        let current = 0;
        const updateCount = () => {
            current += increment;
            if (current < target) {
                stat.textContent = Math.ceil(current);
                requestAnimationFrame(updateCount);
            } else {
                stat.textContent = target;
            }
        };
        updateCount();
    });
});

// 添加简单的淡入动画样式到页面
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .dark-mode {
        background-color: #1a1a1a;
        color: #f0f0f0;
    }
    .dark-mode .paper-card, .dark-mode .overview-card, .dark-mode .toc-card {
        background-color: #2d2d2d;
        border-color: #404040;
        color: #fff;
    }
    .dark-mode .hero {
        background: linear-gradient(135deg, #000000 0%, #1a237e 100%);
    }
`;
document.head.appendChild(style);