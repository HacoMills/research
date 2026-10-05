# ============================================================
#  quant 文件夹一键整理 (只移动, 不删除任何文件)
#
#  用法: 在 quant 文件夹里右键 → "在终端中打开", 运行:
#     powershell -ExecutionPolicy Bypass -File .\organize.ps1 -DryRun   # 先预览, 不真的移动
#     powershell -ExecutionPolicy Bypass -File .\organize.ps1           # 正式执行
#
#  规则: 目标位置已有同名文件时跳过, 不覆盖; 可以重复运行
# ============================================================
param([switch]$DryRun)

$root = $PSScriptRoot
Set-Location $root
$moved = 0; $skipped = 0

function Ensure-Dir($p) {
    if (-not (Test-Path $p)) {
        if ($DryRun) { Write-Host "  [预览] 新建文件夹 $p" -ForegroundColor DarkGray }
        else { New-Item -ItemType Directory -Path $p -Force | Out-Null }
    }
}

function Move-One($src, $dstDir, $newName = $null) {
    if (-not (Test-Path -LiteralPath $src)) { return }
    $name = if ($newName) { $newName } else { Split-Path $src -Leaf }
    $dst = Join-Path $dstDir $name
    if (Test-Path -LiteralPath $dst) {
        Write-Host "  跳过 (目标已存在): $src" -ForegroundColor Yellow
        $script:skipped++
        return
    }
    Ensure-Dir $dstDir
    if ($DryRun) { Write-Host "  [预览] $src  →  $dst" -ForegroundColor Cyan }
    else { Move-Item -LiteralPath $src -Destination $dst; Write-Host "  $src  →  $dst" -ForegroundColor Green }
    $script:moved++
}

function Move-Glob($dir, $pattern, $dstDir, $exclude = @()) {
    if (-not (Test-Path $dir)) { return }
    Get-ChildItem -LiteralPath $dir -File -Filter $pattern | Where-Object { $exclude -notcontains $_.Name } |
        ForEach-Object { Move-One $_.FullName $dstDir }
}

function Remove-IfEmpty($dir) {
    if ((Test-Path $dir) -and -not (Get-ChildItem -LiteralPath $dir -Force)) {
        if ($DryRun) { Write-Host "  [预览] 删除空文件夹 $dir" -ForegroundColor DarkGray }
        else { Remove-Item -LiteralPath $dir; Write-Host "  删除空文件夹 $dir" -ForegroundColor DarkGray }
    }
}

if ($DryRun) { Write-Host "`n=== 预览模式: 不会移动任何文件 ===`n" -ForegroundColor Magenta }

$turtle = "strategies\turtle"

Write-Host "`n[1] 原始数据 → data\raw\"
Move-One "data\securities" "data\raw"
foreach ($f in "btc_usdt_15m_2022_to_now.csv", "df_final_prepared_v5.csv", "okx_btc_funding_rate_v4.csv") {
    Move-One "data\$f" "data\raw"
}

Write-Host "`n[2] 币圈缓存 → data\cache\okx\"
Move-Glob "data\cache" "*.csv" "data\cache\okx"
Move-Glob "$turtle\cache" "*.csv" "data\cache\okx"
Remove-IfEmpty "$turtle\cache"

Write-Host "`n[3] 旧版脚本 → strategies\turtle\archive\"
foreach ($f in "turtle_bars_backtest.py", "turtle_multi_period_backtest.py", "turtle_dashboard.py") {
    Move-One "$turtle\$f" "$turtle\archive"
}
Move-One "$turtle\Claude outputs" "$turtle\archive" "claude_outputs"
Move-One "$turtle\readme" "$turtle\archive" "readme_old.txt"

Write-Host "`n[4] 旧报告和图片 → reports\turtle\ 、figures\turtle\"
Move-Glob "$turtle" "*.md" "reports\turtle" @("README.md")
Move-Glob "$turtle" "turtle_screener_*.csv" "reports\turtle"
Move-Glob "$turtle" "*.png" "figures\turtle"
Move-Glob "reports" "*.md" "reports\turtle"
Remove-IfEmpty "$turtle\charts"

Write-Host "`n[5] 分层框架上线后, 旧的回测脚本 → strategies\turtle\archive\ (新入口: 在 quant 根目录 python -m backtest <配置>)"
foreach ($f in "turtle_timeframe_backtest.py", "turtle_multi_asset.py", "turtle_cross_asset.py",
               "turtle_engine.py", "turtle_analytics.py", "run.py") {
    Move-One "$turtle\$f" "$turtle\archive"
}

Write-Host "`n完成: 移动 $moved 项, 跳过 $skipped 项" -ForegroundColor Magenta
if ($DryRun) { Write-Host "确认无误后, 去掉 -DryRun 再运行一次`n" -ForegroundColor Magenta }
