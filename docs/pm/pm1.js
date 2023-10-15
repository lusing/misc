function mindmap_to_opml(mindmap_str) {
    // 将 mindmap 字符串转换为 DOM 树。
    const mindmap_tree = new DOMParser().parseFromString(mindmap_str, "text/xml");
  
    // 创建 OPML 文档。
    const opml = document.createElement("opml");
    opml.setAttribute("version", "1.0");
  
    // 遍历 mindmap 树，将每个节点转换为 OPML 节点。
    for (const node of mindmap_tree.querySelectorAll("node")) {
      const outline = document.createElement("outline");
      outline.setAttribute("type", "rss");
      outline.setAttribute("title", node.getAttribute("text"));
  
      // 递归处理子节点。
      for (const child of node.querySelectorAll("node")) {
        const sub_outline = mindmap_to_opml(child.textContent);
        outline.appendChild(sub_outline);
      }
  
      opml.appendChild(outline);
    }
  
    // 返回 OPML 文档。
    return opml.outerHTML;
  }
  
  // 测试代码。
  const mindmap_str = `
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
  `;
  
  const opml_str = mindmap_to_opml(mindmap_str);
  
  console.log(opml_str);
  