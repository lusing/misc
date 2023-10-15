import xml.etree.ElementTree as ET

def mindmap_to_opml(mindmap_str):
  """
  将mindmap格式字符串转换成opml格式字符串。

  Args:
    mindmap_str: mindmap格式字符串。

  Returns:
    opml格式字符串。
  """

  mindmap_tree = ET.fromstring(mindmap_str)
  root = ET.Element("opml")
  root.attrib["version"] = "1.0"
  for node in mindmap_tree.findall("node"):
    outline = ET.SubElement(root, "outline")
    outline.attrib["type"] = "rss"
    outline.attrib["title"] = node.attrib["text"]
    for child in node.findall("node"):
      sub_outline = ET.SubElement(outline, "outline")
      sub_outline.attrib["type"] = "rss"
      sub_outline.attrib["title"] = child.attrib["text"]
  return ET.tostring(root, encoding="utf-8").decode("utf-8")


if __name__ == "__main__":
  mindmap_str = """
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
      团队
        关键点
          协作的项目团队涉及的因素
            团队共识
            组织结构
            过程
          协作的项目团队文化
            职权
            担责
            职责
      干系人
        关键点
          干系人参与的重要性
          有效果且有效率地参与和沟通
      价值
        关键点
          项目价值
          关注预期成果
      系统思考
        关键点
          将系统整体性思维应用于项目
          将系统整体性思维应用于项目团队
          识别、评估和响应系统交互带来的收益
      领导力
        关键点
          有效领导力
          领导力风格
          领导力技能的培养
      裁剪
        关键点
          裁剪的重要性
          裁剪的收益
      质量
        关键点
          质量的内容
          质量的测量
          质量的收益
      复杂性
        关键点
          复杂性的来源
            人类行为
            系统行为
            不确定性和模糊性
            技术创新
      风险
        关键点
          风险及应对方法
      适应性和韧性
        关键点
          适应性和韧性
          提升项目团队的适应性和韧性的能力
      变革
        关键点
          积极驱动变革
  """
  opml_str = mindmap_to_opml(mindmap_str)
  print(opml_str)
  