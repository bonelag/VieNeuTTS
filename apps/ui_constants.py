import gradio as gr

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
.container { max-width: 1400px; margin: auto; }
/* Compact control rows (Voice Cloning → saved voices): caption above, one-line
   controls vertically centred so a small button sits level with the dropdown. */
.field-caption { margin: 4px 0 -6px 0; }
.field-caption p { margin: 0; font-size: 0.85rem; color: var(--block-title-text-color); }
.inline-row { align-items: center; }
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    color: white !important;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white !important;
}
.gradient-text {
    background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-icon {
    color: white;
}
.status-box {
    font-weight: 500;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(99, 102, 241, 0.03);
    border-radius: 8px;
}
.status-box textarea {
    text-align: center;
    font-family: inherit;
}
.estimate-box {
    font-weight: 500;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(99, 102, 241, 0.03);
    border-radius: 8px;
}
.estimate-box textarea {
    text-align: center;
    font-family: inherit;
}
.model-card-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 15px;
    font-size: 0.9rem;
    text-align: center;
    color: white !important;
}
.model-card-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: white !important;
}
.model-card-item strong {
    color: white !important;
}
.model-card-item span {
    color: white !important;
}
.model-card-link {
    color: #60A5FA;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
.model-card-link:hover {
    color: #22D3EE;
    text-decoration: underline;
}
.warning-banner {
    background-color: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}
.warning-banner-title {
    color: #92400e;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.warning-banner-grid {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}
.warning-banner-item {
    flex: 1;
    min-width: 240px;
    background: #fef3c7;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #fde68a;
}
.warning-banner-item strong {
    color: #b45309;
    display: block;
    margin-bottom: 4px;
    font-size: 0.95rem;
}
.warning-banner-content {
    color: #78350f;
    font-size: 0.9rem;
    line-height: 1.5;
}
.warning-banner-content b {
    color: #451a03;
    background: rgba(251, 191, 36, 0.2);
    padding: 1px 4px;
    border-radius: 4px;
}
.script-box textarea {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
.speaker-table {
    margin-top: 10px;
}

/* Hide number spinner up/down arrows */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}
input[type=number] {
    -moz-appearance: textfield !important;
    appearance: textfield !important;
}

/* Seed control row styling */
.seed-control-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
    position: relative !important;
}
#seed_input_box {
    position: relative !important;
}
.seed-input-wrapper {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}
.seed-input-wrapper input,
#seed_input_box input {
    padding-right: 34px !important;
}
#btn_reset_seed,
.seed-clear-btn {
    position: absolute !important;
    right: 8px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    line-height: 1 !important;
    cursor: pointer !important;
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    min-height: 22px !important;
    max-width: 22px !important;
    max-height: 22px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 50% !important;
    transition: all 0.15s ease !important;
    z-index: 10 !important;
}
#btn_reset_seed:hover,
.seed-clear-btn:hover {
    color: #ef4444 !important;
    background: rgba(239, 68, 68, 0.2) !important;
}
#btn_save_seed,
.seed-save-btn {
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    min-width: 48px !important;
    max-width: 68px !important;
    margin-bottom: 15px !important;
    padding: 0 8px !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    border-radius: 6px !important;
}

/* Audio stats card (Screenshot 3 style) */
.audio-stats-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 8px;
    text-align: left;
    font-family: inherit;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.stats-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #4ade80;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.stats-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
}
.stats-list {
    list-style-type: circle;
    padding-left: 22px;
    margin: 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.85;
}
.stats-list li {
    margin-bottom: 4px;
}
.stats-badge {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.3);
    padding: 2px 8px;
    border-radius: 5px;
    color: #e2e8f0;
    font-weight: 500;
}
.stats-seed {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 1rem;
    color: #38bdf8;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 2px 8px;
    border-radius: 5px;
    font-weight: 600;
}
.inline-use-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #ffffff !important;
    border: 1px solid #60a5fa;
    border-radius: 5px;
    padding: 2px 10px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    margin-left: 10px;
    vertical-align: middle;
    transition: all 0.15s ease-in-out;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.inline-use-btn:hover {
    background: #1d4ed8;
    transform: scale(1.05);
}
.status-live-msg {
    padding: 10px 14px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.95rem;
    line-height: 1.5;
}
.hidden-bridge {
    display: none !important;
}
"""

head_html = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
<script>
window.useSeed = function(seed) {
    var inputs = document.querySelectorAll('#seed_input_box input, .no-spinner-input input');
    for (var i = 0; i < inputs.length; i++) {
        inputs[i].value = seed;
        inputs[i].dispatchEvent(new Event('input', { bubbles: true }));
        inputs[i].dispatchEvent(new Event('change', { bubbles: true }));
    }
    var cache = document.querySelector('#seed_cache_box textarea, #seed_cache_box input');
    if (cache) {
        cache.value = seed;
        cache.dispatchEvent(new Event('input', { bubbles: true }));
        cache.dispatchEvent(new Event('change', { bubbles: true }));
    }
    var btn = document.getElementById('btn_hidden_use_seed');
    if (btn) {
        setTimeout(function() { btn.click(); }, 30);
    }
};

function setupSeedUI() {
    var seedBox = document.getElementById('seed_input_box');
    var clearBtn = document.getElementById('btn_reset_seed');
    if (!seedBox || !clearBtn) return;

    var input = seedBox.querySelector('input');
    if (!input) return;

    var parent = input.parentElement;
    if (!parent.classList.contains('seed-input-wrapper')) {
        var wrapper = document.createElement('div');
        wrapper.className = 'seed-input-wrapper';
        parent.insertBefore(wrapper, input);
        wrapper.appendChild(input);
        wrapper.appendChild(clearBtn);
    }
}
setInterval(setupSeedUI, 250);
</script>
"""

DEFAULT_TEXT_GPU = "Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà."
DEFAULT_TEXT_TURBO = (
    "Trước đây, hệ thống điện chủ yếu sử dụng direct current, nhưng Tesla đã chứng minh rằng alternating current is more efficient for long-distance transmission. Nhờ đó, điện có thể được truyền đi xa hơn với ít tổn thất năng lượng hơn. Đây là một bước tiến cực kỳ quan trọng trong ngành điện.\n\n"
    "Một trong những phát minh nổi tiếng của ông là Tesla coil, một thiết bị có thể tạo ra điện áp rất cao và những tia sét nhân tạo. This device is still used today in demonstrations và trong một số ứng dụng nghiên cứu. Khi nhìn thấy những tia điện này, nhiều người cảm thấy vừa ấn tượng vừa hơi đáng sợ."
)

# v3 Turbo demo text — khoe giọng tự nhiên + tag cảm xúc [cười] (tính năng mới, thử nghiệm).
DEFAULT_TEXT_V3 = (
    "Xin chào mọi người! [hắng giọng] Như bạn đang nghe thấy đấy, tốc độ xử lý của mình cực kỳ nhanh và mượt mà, giúp phản hồi gần như ngay lập tức theo thời gian thực. Chính vì vậy, mình rất phù hợp để ứng dụng trực tiếp vào các hệ thống Chatbot thông minh, trợ lý ảo, hoặc làm tổng đài viên tự động cho các doanh nghiệp. Tiện lợi quá đúng không ạ? [cười] Hi vọng phiên bản nâng cấp v3 này sẽ mang lại trải nghiệm tuyệt vời cho dự án của bạn."
)
