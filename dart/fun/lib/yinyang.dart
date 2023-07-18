class YinYang {
    int value=0;
    static const int Yin = 0;
    static const int Yang = 1;
    YinYang(int val) {
        this.value = val;
    }
    bool isYin() {
        return value % 2 == Yin;
    }
    bool isYang() {
        return value % 2 == Yang;
    }
    String getName() {
        return isYin() ? "阴" : "阳";
    }
}