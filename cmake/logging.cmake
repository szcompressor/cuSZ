if(PSZ_LOG_DBG_MODE)

add_compile_definitions(PSZ_DBG_ON)

endif()

if(PSZ_LOG_SANITIZE_MODE)

add_compile_definitions(PSZ_SANITIZE_ON)

endif()


if(PSZLOG_ENABLE_ALL)

add_compile_definitions(PSZ_DBG_ON)
add_compile_definitions(PSZ_SANITIZE_ON)

endif()