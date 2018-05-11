if [ -f scan2.py ] && [ -f scan_std.py ]; then
	echo "error -> both file exists"
elif [ ! -f scan2.py ] && [ -f scan_std.py ]; then
	echo "scan.py scan_std.py -> scan2.py"
elif [ -f scan2.py ] && [ ! -f scan_std.py ]; then
	echo "scan.py scan2.py -> scan_std.py"
else
	echo "error -> both file not exists"
fi
