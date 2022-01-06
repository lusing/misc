const ua = window.navigator.userAgent;
/**
 * judge is true
 *
 * @param {*} val
 * @returns Boolean
 */

export const isTrue = val => {
  return val === 'true' || val === true;
};
/**
 * 打开新的页面
 * @param {String} url 页面 url
 * @returns {Undefined} undefined
 */

export function openUrl(url) {
  if (url) {
    if (window.WindVane && window.WindVane.call) {
      window.WindVane.call('WVNative', 'openWindow', {
        url
      }, () => {}, () => {
        location.href = url;
      });
    } else {
      location.href = url;
    }
  }
}
/** 商品状态颜色格式化
 *  @param  {Object}  item  商品数据对象
 *  @param  {Object}  style 需要合并的样式数据
 *  @return {Object}        格式化后的样式对象
 */

export const formatStatusColor = (item, style) => {
  style = style || {}; // 预热期状态

  if (item.sellStatus === '1') {
    return { ...style,
      color: '#00BF7F'
    };
  } // 预售期状态


  if (item.sellStatus === '2') {
    return { ...style,
      color: '#A82CFF'
    };
  } // 正式期状态


  return style;
}; // 将秒转化为分秒

export const formateSeconds = endTime => {
  let secondTime = parseInt(endTime); // 将传入的秒的值转化为Number

  let min = 0; // 初始化分

  let result = '';

  if (secondTime > 60) {
    // 如果秒数大于60，将秒数转换成整数
    min = parseInt(secondTime / 60); // 获取分钟，除以60取整数，得到整数分钟

    secondTime = parseInt(secondTime % 60); // 获取秒数，秒数取佘，得到整数秒数
  } // result = `${min.toString().padStart(2, '0')}:${secondTime.toString().padStart(2, '0')}`;


  result = `${min < 10 ? `0${min.toString()}` : min}: ${secondTime < 10 ? `0${secondTime.toString()}` : secondTime}`;
  return result;
};
/**
 * 获取剩余时间
 * @param  {Number} countDown 倒计时
 * @return {Object}           剩余时间对象
 */

export const getCountDownTime = countDown => {
  const t = countDown * 1;
  const seconds = Math.floor(t % 60);
  const minutes = Math.floor(t / 60 % 60);
  const hours = Math.floor(t / (60 * 60) % 24);
  const days = Math.floor(t / (60 * 60 * 24));
  return {
    total: t < 0 || Number.isNaN(t) ? 0 : t,
    days: days < 0 || Number.isNaN(days) ? 0 : days,
    hours: hours < 0 || Number.isNaN(hours) ? 0 : hours,
    minutes: minutes < 0 || Number.isNaN(minutes) ? 0 : minutes,
    seconds: seconds < 0 || Number.isNaN(seconds) ? 0 : seconds
  };
}; // 系统判断

let os = '';

if (/iPhone|iPod|iPad|iOS/i.test(ua)) {
  os = 'ios';
} else if (/Android/i.test(ua)) {
  os = 'android';
}

export const appOs = os;
let isLowPerformancePhoneCache = '';
export const isLowPerformancePhone = () => {
  if (isLowPerformancePhoneCache !== '') {
    return Promise.resolve(isLowPerformancePhoneCache);
  }

  return getDeviceLevel().catch(e => {
    isLowPerformancePhoneCache = false;
    return isLowPerformancePhoneCache;
  });
};
const lowPerformancePhoneSDKBizID = '2020juhuasuan'; // 文档 https://yuque.antfin-inc.com/lg2g7v/yg1ggp/qomvxq

const getDeviceLevel = () => {
  if (!window.WindVane || !window.WindVane.isAvailable) {
    console.log('WindVane is no Available');
    return Promise.resolve(false);
  }

  return new Promise(resolve => {
    window.WindVane.call('AliHADowngradeSDKBridge', 'getDowngradeInfo', {
      BizID: lowPerformancePhoneSDKBizID
    }, e => {
      const func = e.result.tactics_function; // 功能策略

      const perf = e.result.tactics_performance; // 性能策略

      if (!isLegality(func, perf)) {
        return resolve(callOldPerformanceInfo());
      }

      isLowPerformancePhoneCache = perf === 'degrade';
      return resolve(isLowPerformancePhoneCache);
    }, e => {
      resolve(callOldPerformanceInfo());
    });
  });
};

function callOldPerformanceInfo() {
  return new Promise((resolve, reject) => {
    window.WindVane.call('AliHADeviceEvaluationBridge', 'getPerformanceInfo', {}, e => {
      isLowPerformancePhoneCache = e.deviceLevel === 3;
      resolve(isLowPerformancePhoneCache);
    }, e => {
      // alert(JSON.stringify(e))
      console.log('callOldPerformanceInfo', e);
      reject(e);
    });
  });
}

function isLegality(func, perf) {
  if (!func || !perf) {
    return false;
  }

  const version = window.lib.env.aliapp && window.lib.env.aliapp.version;

  if (!version || version.gte('8.11.0') && version.lte('8.11.1')) {
    return false;
  }

  return true;
}
/**
 * 节流函数
 * @param {Function} func 回调函数
 * @param {Number} interval 间隔时间ms
 * @return {Function} func
 */


export const throttle = (func, interval = 300) => {
  if (typeof func !== 'function') return;
  let canRun = true;
  return function () {
    if (!canRun) return;
    func.apply(this, arguments);
    canRun = false;
    setTimeout(() => {
      canRun = true;
    }, interval);
  };
};
export const getNetworkType = fun => {
  window.WindVane && window.WindVane.isAvailable && window.WindVane.call('WVNetwork', 'getNetworkType', {}, e => {
    // alert(JSON.stringify(e));
    fun(e);
  });
}; // /**
//  * jsTracker 上报
//  * @param {string} msg 上报消息
//  * @param {string} urlType 划分纬度
//  * @param {number} sampling 采样率， 1 => 100%  10 => 10%  100 => 1%
//  * @return {Undefined} undefined
//  */
// export const jsTracker = (msg, urlType, sampling = 1) => {
//   // 依赖于 <script src="//g.alicdn.com/tb/tracker/index.js">
//   window.JSTracker2.push({
//     msg,
//     type: 'custom',
//     url: urlType,
//     sampling,
//   });
// };

let _appName = '';
let _appVersion = '';
const matched = ua.match(/AliApp\(([a-zA-Z-]+)\/([\d.]+)\)/);

if (matched) {
  _appName = matched[1]; // TB TM

  _appVersion = matched[2];
}

export const appName = _appName;
export const appVersion = _appVersion; // 是否是 pha 环境
// ios 961 到 970 默认为 pha 环境，避免出现双底 bar 问题

let _isIOS961to970 = false; // 包含 961 不包含 970

if (os === 'ios' && _appVersion && versionStringCompare(_appVersion, '9.6.1') >= 0 && versionStringCompare(_appVersion, '9.7.0') < 0) {
  _isIOS961to970 = true;
}

export const isPHA = typeof window.__pha_environment__ === 'object' || _isIOS961to970; // 保留排查 9.6.1 版本刷新后环境判断问题
// // 是否开启底部导航兜底
// let _enableBottomBar = !isPHA && _appName === 'TB';
// if (process.env.CDN_ENV !== 'production') {
//   if (window.__queryvalue.enableBottomBar === 'true') {
//     _enableBottomBar = true;
//   }
// }
// export const enableBottomBar = _enableBottomBar;
// 版本比对

export function versionStringCompare(preVersion = '', lastVersion = '') {
  const sources = preVersion.split('.');
  const dests = lastVersion.split('.');
  const maxL = Math.max(sources.length, dests.length);
  let result = 0;

  for (let i = 0; i < maxL; i++) {
    const preValue = sources.length > i ? sources[i] : 0;
    const preNum = Number.isNaN(Number(preValue)) ? preValue.charCodeAt() : Number(preValue);
    const lastValue = dests.length > i ? dests[i] : 0;
    const lastNum = Number.isNaN(Number(lastValue)) ? lastValue.charCodeAt() : Number(lastValue);

    if (preNum < lastNum) {
      result = -1;
      break;
    } else if (preNum > lastNum) {
      result = 1;
      break;
    }
  }

  return result;
}
export const historyBack = () => {
  if (isPHA) {
    window.WindVane && window.WindVane.call('PHAJSBridge', 'navigationBar.back', {}, e => {
      console.log(e);
    }, e => {
      console.log(e);
      history.back();
    });
  } else {
    window.WindVane && window.WindVane.call('WebAppInterface', 'pop', {}, () => {}, e => {
      // alert('failure: ' + JSON.stringify(e));
      console.log(e);
      history.back();
    });
  }
}; // 获取屏幕高度

export const basicConfig = {
  WINDOWHEIGHT: `${window.screen.height}px`
};
/**
 * 截取字符串长度
 * @param {string} str 字符串
 * @param {number} length 要截取的长度
 * @returns 截取后的 字符串...
 */

export const interceptString = (str = '', length = 0) => {
  let newStr = '';

  if (str.length > length) {
    newStr = `${str.substr(0, length)}...`;
  } else {
    newStr = str;
  }

  return newStr;
}; // 时间格式替换

export const formatDate = changeDate => {
  const mon = new Date(changeDate).getMonth() + 1;
  const day = new Date(changeDate).getDate();
  return `${mon}月${day}日`;
};
export const formatDateAndHour = f => {
  const t = new Date(f);
  const mon = t.getMonth() + 1;
  const day = t.getDate();
  const hour = t.getHours();
  return `${mon}月${day}日${hour}时`;
};
export const checkWechatEnv = () => {
  return new Promise((resolve, reject) => {
    const UA = window.navigator.userAgent.toLowerCase();

    if (UA.match(/MicroMessenger/i) == 'micromessenger' || UA.match(/_SQ_/i) == '_sq_') {
      resolve(true);
    } else {
      reject(false);
    }
  });
};
export const setTimeoutPromise = (callback, ts) => {
  return new Promise(resolve => {
    setTimeout(() => {
      callback();
      resolve();
    }, ts);
  });
};
