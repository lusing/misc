function mindmapToOpml(mindmap) {

    let opml = `<opml version="1.0">\n`;
    
    // 获取标题
    const title = mindmap.split('\n')[0].trim();
    opml += `<head>\n<title>${title}</title>\n</head>\n`;
    
    opml += '<body>\n';
  
    // 分行处理
    const lines = mindmap.split('\n');
  
    // 用于表示当前层级
    let level = 0;
  
    for (let i = 1; i < lines.length; i++) {
  
      const line = lines[i].trim();
  
      // 匹配标题
      const matchTitle = line.match(/^[^\s].*/);
      if (matchTitle) {
        const title = matchTitle[0];
        opml += `${'\t'.repeat(level)}<outline text="${title}">\n`;
        level++;
        continue;
      }
  
      // 匹配子节点
      const matchSub = line.match(/^\s+- (.*)/);
      if (matchSub) {
        const subTitle = matchSub[1];
        opml += `${'\t'.repeat(level)}<outline text="${subTitle}"/>\n`;
        continue;
      }
  
      // 匹配空行,表示上一层结束
      if (!line) {
        level--;
        opml += `${'\t'.repeat(level)}</outline>\n`;
      }
  
    }
  
    opml += '</body>\n</opml>';
  
    return opml;
  
  }
  
  // 测试
  const mindmap = `
mindmap
  
    标题1
    
        子标题1
    
        子标题2
  
    标题2
  
  `;
  
  console.log(mindmapToOpml(mindmap));
