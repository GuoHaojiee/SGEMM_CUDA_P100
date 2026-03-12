.PHONY: all build debug clean profile bench

CMAKE    := cmake
BUILD_DIR      := build
BENCHMARK_DIR  := benchmark_results

# 默认目标：Release 编译
all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR) --no-print-directory

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR) --no-print-directory

clean:
	@rm -rf $(BUILD_DIR)

# 运行所有内核 benchmark 并生成图表（需要 matplotlib）
bench: build
	@python3 benchmark.py

# Nsight Compute 性能剖析
# 用法: make profile KERNEL=<内核编号>  例: make profile KERNEL=10
profile: build
	@mkdir -p $(BENCHMARK_DIR)
	ncu --set full --export $(BENCHMARK_DIR)/kernel_$(KERNEL) \
	    --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

# 查看内核 PTX / SASS（P100 = sm_60）
# 用法: make sass KERNEL_FUNC=<函数名子串>  例: make sass KERNEL_FUNC=Warptiling
KERNEL_FUNC ?= Warptiling
SASS_FUNC   := $$(cuobjdump -symbols $(BUILD_DIR)/sgemm | grep -i $(KERNEL_FUNC) | awk '{print $$NF}')

sass: build
	cuobjdump -arch sm_60 -sass -fun $(SASS_FUNC) $(BUILD_DIR)/sgemm | c++filt > $(BUILD_DIR)/kernel.sass
	cuobjdump -arch sm_60 -ptx  -fun $(SASS_FUNC) $(BUILD_DIR)/sgemm | c++filt > $(BUILD_DIR)/kernel.ptx
	@echo "输出：$(BUILD_DIR)/kernel.sass  $(BUILD_DIR)/kernel.ptx"
