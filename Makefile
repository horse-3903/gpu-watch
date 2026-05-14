INSTALL_DIR := /usr/local/bin
SRC_SCRIPT  := src/gpu-watch
SRC_METRICS := src/gpu_watch_metrics.py

.PHONY: install uninstall deb clean

install:
	install -m 755 $(SRC_SCRIPT)  $(INSTALL_DIR)/gpu-watch
	install -m 755 $(SRC_METRICS) $(INSTALL_DIR)/gpu_watch_metrics.py
	@echo "Installed gpu-watch to $(INSTALL_DIR)"

uninstall:
	rm -f $(INSTALL_DIR)/gpu-watch
	rm -f $(INSTALL_DIR)/gpu_watch_metrics.py
	@echo "Uninstalled gpu-watch"

deb:
	chmod +x build_deb.sh
	./build_deb.sh

clean:
	rm -rf build/ dist/
	@echo "Cleaned build artifacts"
