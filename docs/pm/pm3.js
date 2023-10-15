function mindmapToOpml(mindmapText) {
    const lines = mindmapText.trim().split('\n');
    let opml = '<opml version="2.0">\n<head>\n<title>Mindmap</title>\n</head>\n<body>\n';
    const stack = [];
    let prevIndent = -1;

    for (let i = 1; i < lines.length; i++) {  // Skip the first line (mindmap)
        const line = lines[i];
        const indent = line.search(/\S/);  // Find the first non-space character
        const text = line.trim();
        
        if (indent > prevIndent) {
            const parent = stack[stack.length - 1] || '';
            stack.push(parent + '<outline text="' + text + '">\n');
        } else if (indent < prevIndent) {
            for (let j = 0; j < (prevIndent - indent) / 4 + 1; j++) {
                opml += stack.pop() + '</outline>\n';
            }
            const parent = stack[stack.length - 1] || '';
            stack.push(parent + '<outline text="' + text + '">\n');
        } else {
            opml += stack.pop() + '</outline>\n';
            const parent = stack[stack.length - 1] || '';
            stack.push(parent + '<outline text="' + text + '">\n');
        }

        prevIndent = indent;
    }

    while (stack.length > 0) {
        opml += stack.pop() + '</outline>\n';
    }

    opml += '</body>\n</opml>';
    return opml;
}

// Mindmap text as a string
const mindmapText = `
mindmap
    项目管理标准
        价值交付系统
            创造价值
                价值交付组件
                信息流
            组织治理系统
            与项目有关的职能
                提供监督和协调
                提出目标和反馈
                引导和支持
                开展工作并贡献洞察
                运用专业知识
                提供业务方向和洞察
                提供资源和方向
                维持治理
            项目环境
                内部环境
                外部环境
            产品管理考虑因素
        项目管理原则
            管家式管理 stewardship
                关键点
                    关注组织内部和外部的职责
                    坚持诚信、关心、可信、合规原则
                    秉持整体观，综合考虑
                        财务
                        社会
                        技术
                        可持续的发展环境
                工作内容
                职责
                    诚信
                    关心
                    可信
                    合规
            团队 team
                关键点
                    项目是由项目团队交付的
                    项目团队通常会建立自己的本地文化
                    协作的项目团队环境有助于
                        与其他组织文化和指南保持一致
                        个人和团队的学习和发展
                        为交付期望成果做出最佳贡献
                协作的项目团队涉及的因素
                    团队共识
                    组织结构
                        
                    过程
                协作的项目团队文化
                    职权
                    担责
                    职责
            干系人 stakeholders
                关键点
                干系人参与的重要性
                有效果且有效率地参与和沟通
            价值 value
                关键点
                项目价值
                关注预期成果
            系统思考 system thinking
                关键点
                将系统整体性思维应用于项目
                将系统整体性思维应用于项目团队
                识别、评估和响应系统交互带来的收益
            领导力 leadership
                关键点
                有效领导力
                领导力风格
                领导力技能的培养
            裁剪 tailoring
                关键点
                裁剪的重要性
                裁剪的收益
            质量 quality
                关键点
                质量的内容
                质量的测量
                质量的收益
            复杂性  complexity
                关键点
                复杂性的来源
                    人类行为
                    系统行为
                    不确定性和模糊性
                    技术创新
            风险 risk
                关键点
                风险及应对方法
            适应性和韧性 adaptability and resiliency
                关键点
                适应性和韧性
                提升项目团队的适应性和韧性的能力
            变革 change
                关键点
                积极驱动变革
`;

// Convert to OPML
const opmlText = mindmapToOpml(mindmapText);
console.log(opmlText);
