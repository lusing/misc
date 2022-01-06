## 冗余信息类

1. 多余的console.log
2. 多余的语句、未使用的变量
3. 多余的注释
4. 多余的空行

附原始素材
> 删
> 删掉
> 去掉
> 可删除
> 无用注释
> 可以删掉
> 无用的代码
> 多余的空行
> 多了一个空行
> 无效日志删下
> 没用的代码去掉
> 注释不用就删掉
> 注释没用就删掉
> 注释可以顺带删了
> 无用的注释就删掉
> 不需要的直接删掉
> 不用的代码删一删
> 无意义的log去掉
> 不需要的代码就删掉吧
> 无用的代码直接删除，如果后续有回叙需求，可以添加一条 commit 记录，使用 git 历史回溯
> 注释可以删掉，有git记录了
> 本次双十一需求需要获取地理位置吗？不要的话，这函数留着是为啥
> ?这段把下面的注释删掉吧，有点影响可读性
> console.log 下次记得去掉
> 变量没用到，没需要赋值，下面也是
> 正式代码禁用 console
> 第二个参数没用到
> 去掉多余入参

> getRegionNameById没看到有用到的地方

```js
import { isFeatureSupport, getChannelLink, getRegionNameById, getChannelLinkWithValue } from 'Utils/consoleConfig'
```

> console
```js
  console.log(props);
```

> 上线前注释尽量删除，如后续可能继续使用，建议备注下
```js
    };
  }
  // componentWillMount() {
  //   this.getList('SHOP_COUPON');
  // }
```

> console移除

```js
  const stateMarketDetail = store.getModelState("marketDetail");

  console.log(marketInfo,'marketInfo');
```

> 用到 NSR 了？
```js
import Nsr from './nsr';
```

> 注释的代码，最好有明确的jsdoc注释标识为啥注视
```js
  },
};
// const getMonth = date => {
//   let month = date.getMonth() + 1;
//   if (month < 10) {
```

## 基础逻辑类

### 判空

> 运营配置的playList有没有可能是零的情况？有没有可能斑马后端返回有问题，导致这些配置数据没有下发到前端的情况？如果是这些情况，这里playList后面可以直接用"."不加处理吗？
> 这里增加判断，只有当!playList && playList.length > 0时才去调用接口
> 要不要加个兼容判断，v && v.image
> 根据handleFormateRule返回的内容，baggageDetail可能还是string，baggageDetail.baggageDetails可能是undefined，有这个风险，建议修改

> @薛舒芳 这行前面增加一行保护性代码，requestParams.data 可能不存在.
> ```js
> requestParams.data = requestParams.data ?? {};
> ```

```js
export const getTabbarData = () => {
  return request('tabbar', (requestParams) => {
    requestParams.data.params = JSON.stringify(requestParams.data);
    return requestParams;
  })
```

> @chunxiao.ycx(夙汐)  为 href 增加保护 `href=''`，避免下文  `href.trim()` 执行报错
> ```js
>   // 如果设定为CLK埋点, expType应为空, 避免命中auto-exp的selector
>   const expType = props['exp-type'] || (trackType === 'CLK' ? '' : 'auto');
>   const { onClick: pressCallBack, href, enableExpandAutoExpParams } = props;
>
>   const gokey = isString(trackInfo) ? trackInfo : '';
> ```

> @薛舒芳 ① lib 前也增加下 `?` 即 `?.lib` ② `||` 改成 `??`
```js
const searchParams = window.lib?.env?.params || {};
```

> 对象判空建议这样写 obj && Object.keys(obj).length === 0 && obj.constructor === Object https://stackoverflow.com/questions/679915/how-do-i-test-for-an-empty-javascript-object

```js
    queryCrowd(crowdsList).then((res:any) => {
      if (JSON.stringify(res) === '{}' || JSON.stringify(res?.data) === '{}') {
        setCrowdId('');
      } else {
```

> 这里加上保护，避免服务端返回数据结构不正确导致前端报错～ res?.daat?.model?.userBlessingComponent?.nickName

```js 
setUserName(res.data.model.userBlessingComponent.nickName)
```

### 类型判断

> 严格等于的时候，boolean是恒不等于数字的，(newRankList?.length % 2 && hasMore) === 0，值都是false的
> 这边list里的item里state的类型，你在interface里写的是any，那么这边你打算怎么确保这个三元判断成功？
> 这个万一不是一个Number呢？

> 尽量不要使用 any -- 在 React.children.map 的场景下，child 可能会是一个字符串或数字的，这时候不能使用 React.cloneElement 的，应该要判断下仅当 React.isValidElement 时使用 cloneElement，即改成:  child => React.isValidElement(child) ? React.cloneElement(child, {form,formData}) : child
>
> ```js
> if (!Array.isArray(content) || content.length === 0)
> ```

```js
  // 判空
  if (content.length === 0) {
    return null
  }
```

### 条件判断

> 这里有8个0，上面一个条件只有7个0，仔细检查一下
> splitModal为false的时候会怎样？
> 这里不要用“==”，永远用“===”
> 不要用中文做全等判断

> if代码块建议用花括号包起来。@ali/universal-user今年就因为if括号的问题出过问题
https://yuque.antfin.com/docs/share/cb02649e-56dc-42e9-9dc9-74ffdc907c88
```js
    if (PageContext.disabled !== void (0)) disabled = PageContext.disabled
    return <div style={style}>
```

> 超过3中情况的if可以用switch
```js
    const changeValue = (value) => {
      let result;
      if (adjustPriceMethod === 'fixedPrice') {
        result = num;
      } else if (adjustPriceMethod === 'prorate') {
```

> @chunxiao.ycx(夙汐)  缩写成 `return typeof obj === 'string'`
```js
	if (typeof obj === 'string') {
		return true
```

> 建议逻辑优化 不要使用过长的三元表达式 常量建议使用枚举，  ===‘1’ 这种较难看出意义的建议有注释

```js
                {
                  ((state.awardType === 'RED_PACKET' && state.awardStatus === 'FINISH') || (state.awardType === 'APPEND' && state.awardStatus !== 'FINISH')) &&
                  (state.awardData?.userTask?.feature?.appendTaskHelperType === '1' ? '邀请今日未打开过淘特的用户' : `再邀${state.awardData?.userTask?.feature?.remainAppendTaskHelpThreshold}人可提现到支付宝`)
                }
```

### 数据类型

> 这里item里的fansCnt你写的type是number，而接口返回的v.fansCnt的type是any，可以直接赋值吗？不需要做好类型转换吗？
> 参数initLines增加类型，写成例如“initLines: string”或者“initLines: number”
> 这边函数return类型也要写明
> 参数写明类型，函数写明返回值类型
> 可以不any嘛 ，233333

> any 换泛型
```typescript
export const  getChildren = (arr: Array<any>, val: string, key: string) => {
 ```

### 字符串

> 这个写法很妙，另外建议也尝试下 String.padStart() 这个方法
https://es6.ruanyifeng.com/#docs/string-methods#%E5%AE%9E%E4%BE%8B%E6%96%B9%E6%B3%95%EF%BC%9ApadStart%EF%BC%8CpadEnd

```js 
const s = `000000${source}`;
return s.substr(s.length - 6);
```

> js里array连成逗号间隔的string不需要用reduce，用join就行了，参考 https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Array/join

```js 
    let targetUserIds = playerList.reduce((acc, cur) => {
      return acc + cur.userId + ',';
    }, '');
```

### 可选链操作符

> 这个值居然这么深，写得很严谨。
> 讨论另外两种可行的相较优雅写法：
> 1. 用try...catch，try块里表达式直接链到底，在catch给兜底值
> 2. 这个仓库的打包配置支持可选链操作符，也就是可以写成 tuigaiqianRule?.baggageInfoVO?.baggageCarry?[0]

```js
      list: tuigaiqianRule && tuigaiqianRule.baggageInfoVO && tuigaiqianRule.baggageInfoVO.baggageCarry && tuigaiqianRule.baggageInfoVO.baggageCarry[0] && tuigaiqianRule.baggageInfoVO.baggageCarry[0].second 
```

### 重复判断

> 上面已经是playerList && playerList.length > 0 &&,为啥这里还需要if (playerList.length > 0) ?
> 此处 actCode 字段的校验是必须的吗？

### 默认值

> 既然avaterUrl，userNick，title都有默认值，那么你自己加的fansCnt，state，userId也写上默认值


### 复杂度

> 这个地方为什么循环里面套循环？这样复杂度是O(n^2)。没有更好的写法吗？
> 这里怎么会有这么多需要深拷贝的情况？

### 深拷贝

> 这里为什么做了次深拷贝？
> ```js
> this.state.List.map((item) => Object.assign({}, item)) 
> ```
> 替换成这样可以吗？
> ```
> [...this.state.List]
> ```

```js 
      const sortedList = this.state.List.map((item) => Object.assign({}, item)).sort(compare);
      return sortedList;
```

### 异步

> await 更明了吧
>
> ```js
>  await dispatch({
>      type: `${NAMESPACE}/fetchSetAclConfig`,
>      payload: {
>        Functions: [data]
>      }
>    });
>
>   dispatch({
>        type: `${NAMESPACE}/fetchSetAclConfig`,
>        payload: {
>          Functions: [orderData]
>        }
>      })
>    });
> ```

```js
  handleSubmit = (data, orderData) => {
    const { dispatch } = this.props;
    dispatch({
      type: `${NAMESPACE}/fetchSetAclConfig`,
```




### 变量命名与拼写错误

> Attation拼错，最好全局改一下Attention
> 这个名命不是很好，这里直接用visible命名就好了吧
> 名字是不是叫copyToClipboard会更好
> 首先，类名是大写开头，其次 这个命名不合理
> 拼写错误，不用的话可以删掉，fail同理
> 函数入参名称和调用接口的名称要区分开来
> 这个变量名和 hook 名字太长了

> 用 type，attr 是属性的意思
```js
OPTIMIZED {已优化}
other {{attr}}}`,
  'openapi_product_line?attr': `{attr, select,
BASE {基础产品线}
DATABASE {数据库产品线}
```

> 为什么要大写...
```js
  // 我的
  mine: {
    MyCenter: (opts) => {
      return request({
        api: 'mtop.alipictures.cfp.portal.individual.centre',
```

### 格式

> @薛舒芳 代码提交前，是不是没有进行过 pretty 格式化？你提交代码的命令是什么？正常来说，代码提交会自动进行格式化的。如果没有格式化，执行 `tnpm run lint:fix` 命令进行格式化。

> 代码的缩进还是要注意的，影响代码的可读性
```js
                clearInterval(this.progressInterval);
                //中转合并下单退票
          if(res.data.dialog){
            this.setState(
              {
```

> perttie 不建议用
```js
    "eslint": "eslint --ext .js,.jsx ./",
    "stylelint": "stylelint \"**/*.{css,scss,less}\"",
    "prettier": "prettier **/* --write"
  },
```

### 代码复用

> 多次出现此逻辑，可以考虑抽象一个函数进行复用。
> 1、这个函数需要提到外面 2、写明函数的返回值类型
> 建议下次下面Tab下的东西写成单独的组件
> 为什么要复制，之前的无法完全复用吗？
> 这块代码是复制的还是新写的？
> 这个也抽出来放在工具函数中
> 序列化接口也可以提取下
> 重复代码，封装方法

> 重复性代码都可以抽成配置数据，从数据渲染，相同结构只留一份
```js
          <div className="Carousel-item-shuang11-content" />
        </div>
        <div className="Carousel-item-new">
          <Header />
          <div className="Carousel-item-content">
```

> 这附近有大量的_mainTableDataSource[0]，可以抽取一个变量出来 const firstRowItem=  _mainTableDataSource[0] 然后从里面读属性，否则相当于每次都要从数组里面去读取，性能会有点影响。

```js 
            value: _mainTableDataSource[0].mailNo,
            status: _mainTableDataSource[0].status,
```

### 数据抽象

> 可以改成从数据来渲染，后面会有计划把数据单独抽出来，到时候只要做数据转换就好了
```js
    return (
      <div className="Professiona-content">
```

### 注释

> 注释不明确，不是好注释

> 这里的逻辑可以加点注释解释一下为啥现在不需要用canChangeCity来判断了
```js
    if (tripTypeItinfo === '单程') {
      this.props.resetNotify();
      onNotify({
```

### 版本号

> @chunxiao.ycx(夙汐)  版本号，从 1.0.0 或者 0.1.0 起步，follow SemVer 规范。

```js
{
  "name": "@ali/hp-tracker-link",
  "version": "0.0.1",
  "description": "hp-tracker-link",
  "main": "lib/index.js",
```

### 模块

> @chunxiao.ycx(夙汐)  require 全部改成 import

```js
'use strict';
let Mod = require('./index.web');
```

> /index 最好还是不要省略。不然容易区分不出来是 utils.js 还是utils/index.js

```js
import { isPre } from '@/utils';
```


## 业务逻辑类

### 传参

> 这个方法调用的地方 没有传任何参数进来 logger.navbarExp({});
> 每个商品卡片点击跳转的是什么地址？仔细看需求！跳转的时候有没有把itemId放在参数里带过去？！

### iOS兼容性

> 这里| 最好换成 %7C，早几天刚遇到手猫 ios 遇到非标准url字符会不响应跳转
>

### 生命周期

> 将清空数据的时机放在搜索激活页隐藏之后，之前是在请求之间清空会出现 有数据->清空数据->请求回来数据的变化导致屏幕闪烁一下

> onunload 是页面卸载的时候调用，不是刚开始调用
```js
  const homeRequest = getApp().request('home').then(homeRes => {
    const result = getDynamicResult(homeRes, 'home');
    onUnload(isShowVideo);
```

> 在useEffect里这个timer需要clear下。
> ```js
>  return () => {
>       clearInterval(timer);
>     };
> ```
> 类似：
> ```js
>  useEffect(() => {
>     const timer = setInterval(() => {
>       setTime(now());
>     }, 1000);
>     return () => {
>       clearInterval(timer);
>     };
>   });
> ```

```js
  const timer = setInterval(
    target, 1000);
```

### 组件和容器

> 员工选择器，后面建议可以使用：https://fusion.alibaba-inc.com/45813/biz/3422
> 由于just for you 是无限加载模块，所以需要将模块都使用cell包裹，不可以使用View统一包起来
> live-kit里有getUrlQuery拿参数，不要自己写了。而且不会变的常量，不需要放在state里

> 富文本的渲染还是比较推荐 fusion-mobile 的富文本组件的 https://fusion.alibaba-inc.com/mobile/component/rich-text?spm=fusion-inc.design-inc-fusion.top-nav.d2mobile.48fc2079w654t6&themeid=10264
```html
    <div
      className={'pegasus-richtext-container'}
      dangerouslySetInnerHTML={{ __html: html || DEFAULT_RICHTEXT_HTML }}
    />
```

### 组件结构

> 第一个问题建议后面持续进行修复，首先 App 组件不应该和 Page 在目录结构上同级，单开一个 pages 目录，页面文件放到这个目录下

```js
import Way1iFrame from '../Way1iFrame';
import UmidTrail from '../UmidTrail';
import WiFiAdress from '../WiFiAdress';
import PrivateRoute from './PrivateRoute';
```


### 闭包

> 需要包一层匿名函数
```js
      onAppear={_sendGoldLog(
        '/uhf_4.shaking_entrance.shaking_entrance_novic_guidance',
        'EXP',
```

### URL处理

> 获取url参数建议用我这个，所有decodeURIComponent地方需要try catch
> ```js
> const getQueryParams = (search = location.search): any => {
>   const queryParams: any = search.substring(1).split('&').reduce((params, str) => {
>     let [key, value = ''] = str.split('=')
>     try {
>       key && (params[key] = decodeURIComponent(value))
>     } catch (e) {
>       // console.log(e)
>     }
>     return params
>   }, {});
>   return queryParams
> }
> ```

```js
const getParams = (url: string,) => {
  const search = url.split('?')
  let searchArr = [];
```

> 可以用first index或者 firstIndexOf(.com)，否则，以下url会出问题：https://www.allylikes.com/sc/2000000990.html?spm=a2g34.home.component0.2.694612e9AMCxU1&ssid=eU7BVv&ext_params=%7B%22scene%22%3A%22Banner%22%7D

```js
    const newUrl = url
      .substr(url.lastIndexOf('com') + 'com'.length + 1)
      .substr(0, 1);
    const newStr = url.replace('https://', '');
```

### 页面回退

> 如果没有URL参数，建议直接走URL回退方案
>
> ```js
> /**
>  * 手淘环境：若 WebView 可以 H5 的 HistoryBack，则回退历史；否则直接退出当前 ViewController/Activity。
>  * 浏览器环境：关闭页面
>  * WindVane API：http://h5.alibaba-inc.com/api/TaobaoClient-API.html#WebAppInterface-pop
>  */
> export function webviewBack() {
>   window.WindVane.call('WebAppInterface', 'pop', {}, () => {}, () => {
>     window.close();
>   });
> }
```

```js
  // 链接无参数，不展示页面
  if (!queryString.auth_code) return;
```

### 事件处理

> 可以直接使用handleInputChange函数
```js
  useEffect(() => {
    if(value){
      onChange(value * radix);
    }
  },[value])
 ```

###  本地存储

> 这里为什么要用本地存储？
```js
    } else {
      localStorage.setItem('thirdMenu', record.name);
      history.push({ pathname: '/community/topic/management', search: '', hash: '#', state: { oneData: record } });
    }
```


### 页面缓存

> 是为了获取上一个页面的缓存吗，应该在 onLoad 里面调用 this.$getNavigateData() 获取存放到 localdata 上，然后再在其他地方使用。
> 如果直接写死这个常量，后续框架调整key值，这里业务可能就会受影响。

```js 
const selcData = Cache.get('NAVIGATE_STORAGE_KEY');
```

### 类库

> 有没有本地跑过代码？这个库有安装吗？检查一下package.json里
>

### 运营可配置

> 这里背景色也做成运营可配置，但是默认这个颜色
>

### 埋点

> 热榜和搜索发现数据返回为空的时候增加埋点

> @wb-gch706066(郭春花) @qiucheng.lbw(秋呈)  坑位 2201埋点重复：d0, d1, d2, d0, d1, d2, d0, ...  需要使用累加器 count （从0算起，或者从1算起）

```js
data-spm={`d${$index}`}
```

> @chunxiao.ycx(夙汐)  click 时，如果有 clickTrackInfo，优先取这个值作为 gokey，否则取 trackInfo

```js
		const clickLogkey = '/hp_tracker_link.module.click'
		if (window && window.goldlog && window.goldlog.record) {
			window.goldlog.record(clickLogkey, 'CLK', gokey, 'GET')
		}
	}, [gokey])
```

> @chunxiao.ycx(夙汐)  模块曝光方法，其实可以不用指定，目前 A+ 默认优先使用的 sendBeacon post 方法，不支持的场景会降级到 Image() get 方法，所以曝光方法的参数，其实完全可以she'q舍弃掉，或者默认使用 POST。
> 扩展阅读：
> 1. `Navigator.sendBeacon()`，https://developer.mozilla.org/zh-CN/docs/Web/API/Navigator/sendBeacon
> 2. https://yuque.antfin.com/aplusjs/docs/lxzbxf#0nwlwe

```js
  // 曝光请求方法
  let expMethod = props['exp-method'];
  if (!isValidExpMethod(expMethod)) {
    expMethod = '';
  }
```



### 举一反三

> 不应该只处理blue，要有通用解法

### 黑名单

> 黑名单缩小下范围，即便是营销表单的配置，其实在资源位预览的时候虽然不能做穿越，但是至少也可以看看一些其他配置项的变更，比如字段映射对不对，上次出的一个问题，就是价格字段关联了一个布尔值
这块再想想，然后在二维码扫码的地方文案补充说下，和商品有关的预览这里是没有月光宝盒穿越的，只能看非穿越相关的配置，大概这个意思

```js
// 商品相关内容需月光宝盒穿越，这里排除:普通圈品、淘特表单、中台表单
const BLACK_SOLUTION_LIST = ['ltaoUnifiedSolution', 'ltaoUnifiedMktNew', 'commonMarketingItem'];
```


### 注意多测测

> 这里的判断是要的，测试的时候注意点
```js
              )}
              {/* (未绑定EIP的实例展示 ｜ 未配置SNAT/DNAT）展示 */}
              {/* {!isOldNatUser && !isFetchSnatAndDnatInfo && isUnbind && ( */}
              {true && (
                <ConfigurationGuide
```

### 降级方案

> containerWidth 取不到需要做个兼容
```js
    <div
      style={{
        width: `${containerWidth}px`,
        backgroundColor: isDarkMode ? '#111' : '#FFF',
      }}
```


### 图片

> 兜底图上面有条线

```js
          uri:
            (state.bannerBg || {}).url ||
            'https://img.alicdn.com/imgextra/i3/O1CN01rYRfVN1Qz73Q7jJ7V_!!6000000002046-2-tps-750-412.png',
        }}
        autoPixelRatio={false}
```


### 库使用beta版本
> 需要 RunCook 发正式版本，线上不允许使用 beta 版本

```json
    "@ali/runcook-framework": "1.6.1-beta.0",
```

> @薛舒芳 既然依赖包已经发布了，这个版本就不对了吧？需引用正式版本~

```js
    "@ali/pcom-splayer": "^0.2.14",
    "@ali/pegasus-document": "^3.0.0",
    "@ali/rax-jhs-bottom-float-block": "1.0.2-beta.1",
    "@ali/rax-picture": "^3.1.2",
    "@ali/runcook-framework": "1.5.0-beta.12",
```

### 超时

> @薛舒芳 超时时间设置未 3000 或 5000 吧
```js
    "ecode": 0,
    "type": "GET",
    "timeout": 2000
  }
}
```

### useCallback

> @chunxiao.ycx(夙汐)  建议用 useCallback 括起来，依赖项 `[gokey]`

```js
  // 2101点击埋点
  const clickTrackInfo = () => {
    const clickLogkey = '/hp_tracker_link.module.click';
    if (window && window.goldlog && window.goldlog.record) {
```

### 投放配置

> 投放计划从投放获取，我们模块里需要在schema里配置接收投放的配置，可以参考：
> schema配置：http://gitlab.alibaba-inc.com/pmod/2020-1111-surpriseball/blob/master/src/schema.json#L5-17
> 模块获取配置：props.data.lafiteConfig 不过里面具体的配置格式需要自己打印log看一下，应该是这样的：http://gitlab.alibaba-inc.com/pmod/2020-1111-surpriseball/blob/master/src/mock.json#L2-8

```js
      const params = {
        "headAwardStrategy": {
          "strategyCode": "",  //投放计划code
          "lafiteChannel": ""  //投放计划channel
        },
```

## 电商业务类

### 坑位

> 商品标题没有时，应该单坑隐藏

### 商品

> 声明下minItems  maxItems  限制每页最多吐出2个商品 http://gitlab.alibaba-inc.com/river/spec/blob/master/JSON-Schema.md?spm=a312q.7764190.0.0.aIOQOt&file=JSON-Schema.md

```js 
          "goods": {
            "title": "商品数据",
            "type": "array",
```

### 会员码

> 这个会员码链接是复用之前的么？新版会员码修改链接了，建议使用最新链接
https://market.m.taobao.com/app/lightshopfe/vip-code/home?wh_weex=true（咨询书黎页面参数）

```js 
my.call('navigateToOutside', {
url: 'https://market.m.taobao.com/app/big-vip/vip-code/pages/index_v4?wh_weex=true&wx_navbar_transparent=true&targetUrl=code_page',
});
```

## 运行环境类

### 容灾与兜底

> 这边finfeeds已经接入自动容灾，不需要传参ceFallbackConf，具体学习一下容灾，咨询栖柒和徐俊晖
> 服务端没有有error message的时候才兜底，默认要展示接口返回的错误信息

>   兜底下没有值的情况
```js
// 渲染级联回显
  handleAddressChange = (value, data, extra) => {
    const address = extra.selectedPath.map(item => item.label).join('/');
    this.setState({ address });
  }
```


### 后端接口

> 找后端验证下，我理解post请求后端从request.body里拿参数 不需要stringify的数据
> post请求还需要stringify吗？

### 预发环境

> 为什么跳到预发？

### 内网地址
> 这个不行，你这个语雀是内网的语雀，外部用户看不了

### 链接错误

> 这个链接是错的，tech.support.vpc.cidr
```js
const ticket = getChannelLink('tech.support.vpc.cidr')
```

###  监控类

> 失败的情况，建议使用 JSTracker 上报一下错误，https://yuque.antfin-inc.com/datax/jstracker

```js
      //console.log('success: ' + JSON.stringify(e));
    }, function (e) {
      //console.log('failure: ' + JSON.stringify(e));
    });
  }
```


## 样式类

### 空容器

> 这边为啥要套一个空的div

```html
                  <View>
                    <View className={styles.actName}>活动信息更新中，敬请期待</View>
                  </View>
```

### 样式类名

> className 小写开头

### 内联样式

> @薛舒芳 既然已经使用外联 index.module.css，为什么这里还有内联的一行样式代码？
```js
    <View style={{ position: 'relative', height: '177rpx' }}>
```

> 同样的问题，有 className 了为啥还要写内联样式。内联样式本身就不是一个最佳实践。

### 溢出

> 这个是超出...，可以直接采用weex2.0写法 ，直接在p标签加上ellipsis
> 这个属性不能删，它的作用flex行式布局时是超过宽度自动折行。
>

### 标签
```
如果这个标签不要样式，直接用<Fragment>或者<>
```
> Picture自闭合

> DOM 里没有内容，可以写成自结束标签
```html
      <BusInfo busInfo={busInfo} type={T_SCAN_ORDER}></BusInfo>
```

### 字体

> 严禁自定义字体，使用统一设置
>

### 单位

> 用rpx，不要用em
> style单位写上

### 能用css的不要用图

> 这里以下的几张图片看尺寸都挺大的，但好像用 css 都能实现
```js
        <View
          style={styles.normal_div_17}
          src={'https://img.alicdn.com/imgextra/i3/O1CN01E3wFc61Iu3iNeCw33_!!6000000000952-2-tps-1500-770.png'}
        >
```

### 遮挡

> @薛舒芳：在跟随状态，会把页头标题遮挡住
```js
      {
        headConfig?.banner_top ? <Picture 
          style={{
            width: 750,
```

## 文档类

### 标题

> 文档 Heading 层级调整了
> 从 ## 二级标题开始
```js
import { Member } from '@ali/xixi-design-member';

# 如何使用
```

### 全角与半角

> @chunxiao.ycx(夙汐) 2处标点符号，逗号改成中文的

```md
@ali/hp-tracker-link 是专用埋点组件, 负责spm埋点 和 标准的算法信息埋点(采用黄金令箭上报曝光和点击行为),减轻业务同学的埋点负担，同时完善埋点数据
```


## 非技术问题类

### 不要照抄

> 这边requestParams需要地理位置和排序吗?像故渊咨询清楚马赫接口是否需要传这几个参数，不要照抄原有参考模块逻辑
> 这边只需要保留本次需求里涉及到的取用商品字段，不要把原有参考模块的逻辑直接全部复制。该删的删掉
> 直接用字符串，不要使用国际化。在理解的情况下复制代码。
> @薛舒芳 下次 copy 代码时，留意下是否未定义的变量。这种问题不应该出

### 需求

> 放弃之前的页面，需要评估是否需要删除之前的页面。先搞清楚为什么要做这个需求
> 有机会和产品聊一下吧，为什么不加字段来校验
>


### 不要一次性提交过多

> 本次代码一次性提交了挺多文件的，后面可以及时提交


## 看不懂的

> 这个是干什么的，能不能带到线上的？
> 这个是什么意义。。。

> 建议加上注释，看不懂😭
```js
// https://aone.alibaba-inc.com/workitem/37837775
const DATE_MAP = {
  1: '21',
  5: '25',
  10: '1',
```

### 值从哪里来的？

> @chunxiao.ycx(夙汐)  isonce 这个值怎么来的？
```js
			ref={linkRef}
			href={safeHref}
			isonce
			{...validGokeyParams}
			{...props}
```

## 表扬

> 很棒。有具体类型，方便阅读
> 这代码设计简洁，逻辑清晰
> 用户细节处理到位，赞一个
> 代码整洁，希望继续保持
> 你的代码写得真棒
> 优秀的设计