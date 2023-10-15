import re

mindmap = """
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
                工作内容
                职责
                    诚信
                    关心
                    可信
                    合规
            团队 team
                关键点
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
"""

def mindmap_to_opml(mindmap):
    opml = "<opml version=\"1.0\">\n"
    opml += "<head>\n<title>{}</title>\n</head>\n".format(re.search(r"^(.*?)\n", mindmap, re.M).group(1))
    opml += "<body>\n"
    
    for line in mindmap.split("\n"):
        if re.match(r"^(?!\s+|\t+).+", line):
            title = re.match(r"^(.*?)\s+", line).group(1)
            opml += "<outline text=\"{}\">\n".format(title) 
        elif re.match(r"^\s+-\s+(.*)", line):
            sub_title = re.match(r"^\s+-\s+(.*)", line).group(1)
            opml += "<outline text=\"{}\"/>\n".format(sub_title)
        elif line.strip() == "":
            opml += "</outline>\n"
            
    opml += "</body>\n</opml>"
    
    return opml

print(mindmap_to_opml(mindmap))
