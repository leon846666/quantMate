"""scripts/quant_analysis.py — 量化交易智能分析命令行工具

用法：
    python scripts/quant_analysis.py 000001.SZ
    python scripts/quant_analysis.py 000001.SZ --days 60
    python scripts/quant_analysis.py 600519.SH --save report.txt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.quant_agent import analyze_stock

def main():
    parser = argparse.ArgumentParser(description="QuantMate智能股票分析工具")
    parser.add_argument("stock_code", help="股票代码 (例: 000001.SZ)")
    parser.add_argument("--days", type=int, default=90, help="分析天数 (默认90天)")
    parser.add_argument("--save", help="保存分析报告到文件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    try:
        if args.debug:
            print(f"🔧 调试模式: 分析股票 {args.stock_code}, 天数: {args.days}")

        print("🚀 启动QuantMate智能分析...")
        print("📊 正在获取数据...")

        analysis = analyze_stock(args.stock_code, args.days)

        print("✅ 分析完成!")
        print(analysis)

        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"\n📄 分析报告已保存到: {args.save}")

    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
        return 1
    except Exception as e:
        print(f"\n❌ 分析过程出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    print("\n✅ 程序正常结束")
    return 0

if __name__ == "__main__":
    main()