add_executable(demo_corr_fpfh corr_fpfh.cpp)
target_link_libraries(demo_corr_fpfh PRIVATE turboreg ${TORCH_LIBRARIES})

# 定义需要复制的 DLL 路径（以 c10.dll 为例，需替换为实际路径）
set(C10_DLL_PATH "D:/envs/anaconda3/py310/lib/site-packages/torch/lib/c10.dll")

# 添加构建后事件：将 DLL 复制到可执行文件的输出目录
if(WIN32)
    add_custom_command(
            TARGET demo_corr_fpfh
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${C10_DLL_PATH}"
            $<TARGET_FILE_DIR:demo_corr_fpfh>
            COMMENT "Copying c10.dll to output directory"
    )
    # 若有其他依赖 DLL，可重复上述步骤
    # 例如复制 LibTorch 的其他 DLL：
    set(TORCH_DLL_PATH "D:/envs/anaconda3/py310/lib/site-packages/torch/lib/torch_cpu.dll")
    add_custom_command(
            TARGET demo_corr_fpfh
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TORCH_DLL_PATH}"
            $<TARGET_FILE_DIR:demo_corr_fpfh>
            COMMENT "Copying torch_cpu.dll to output directory"
    )
endif()