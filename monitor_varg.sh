#!/bin/bash

# V.A.R.G Monitoring Script
echo "📊 V.A.R.G System Monitor"
echo "========================"

# Check service status
echo "🔍 Service Status:"
sudo systemctl status varg.service --no-pager -l

echo ""
echo "📈 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2 * 100.0)}')"
echo "Temperature: $(vcgencmd measure_temp | cut -d'=' -f2)"

echo ""
echo "📋 Recent Logs:"
sudo journalctl -u varg.service --no-pager -n 10

echo ""
echo "🔧 Control Commands:"
echo "  Start:   sudo systemctl start varg.service"
echo "  Stop:    sudo systemctl stop varg.service"
echo "  Restart: sudo systemctl restart varg.service"
echo "  Logs:    sudo journalctl -u varg.service -f"
