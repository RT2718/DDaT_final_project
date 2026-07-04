/**
  ******************************************************************************
  * @file    main.c
  * @brief   Acoustic Tracking Motor Control (Goto Logic with Robust RTOS-style ISR)
  ******************************************************************************
  */

#include "main.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MyFlagInterruptHandler(void);
static void MX_USART2_UART_Init(void);
void Error_Handler(uint16_t error);
void Process_Motor_Control(void);

UART_HandleTypeDef huart2;

/* --- Mechanical Conversions --- */
const float STEPS_PER_DEG_PAN = 3345.0f / 360.0f;
const float STEPS_PER_DEG_TILT = 16576.0f / 360.0f;

const float RAD_PER_STEP_PAN = (M_PI / 180.0f) / (3345.0f / 360.0f);
const float RAD_PER_STEP_TILT = (M_PI / 180.0f) / (16576.0f / 360.0f);
const float K_pa = 80.0f;
const float K_pb = 100.0f;
const float K_ia = 30.0f;
const float K_ib = 15.0f;
const float K_da = 32.0f;
const float K_db = 30.0f;
const uint16_t maxSpeed_pan = 1000;
const uint16_t maxSpeed_tilt = 2000;


// 5 degree deadband in radians
const float DEADBAND_RAD = 0.0f * (M_PI / 180.0f);

/* --- Communication Globals --- */
volatile float target_alpha = 0.0f; // Pan Target (from Python 'phi')
volatile float target_beta = 0.0f;  // Tilt Target (from Python 'theta')
volatile uint32_t last_msg_time = 0;

/* --- UART Double Buffering (Ping-Pong) Variables --- */
volatile uint8_t string_ready = 0;
volatile uint8_t write_idx = 0;  // ISR writes to this index
char rx_buf[2][32];              // Two distinct buffers
uint8_t rx_byte;
uint8_t rx_index = 0;


int main(void)
{
    HAL_Init();
    SystemClock_Config();

    /* Initialize Serial Port (USB) */
    MX_USART2_UART_Init();

    /* 1. Initialize Motors (0 = Pan, 2 = Tilt) */
    BSP_MotorControl_SetNbDevices(BSP_MOTOR_CONTROL_BOARD_ID_L6474, 3);
    BSP_MotorControl_Init(BSP_MOTOR_CONTROL_BOARD_ID_L6474, NULL);
    BSP_MotorControl_Init(BSP_MOTOR_CONTROL_BOARD_ID_L6474, NULL);
    BSP_MotorControl_Init(BSP_MOTOR_CONTROL_BOARD_ID_L6474, NULL);

    BSP_MotorControl_AttachFlagInterrupt(MyFlagInterruptHandler);
    BSP_MotorControl_AttachErrorHandler(Error_Handler);

    /* Base Configurations */
    BSP_MotorControl_SetMinSpeed(0, 200);
    BSP_MotorControl_SetMaxSpeed(0, maxSpeed_pan);
    BSP_MotorControl_SetAcceleration(0, 1500);
    BSP_MotorControl_SetDeceleration(0, 1500);

    BSP_MotorControl_SetMinSpeed(2, 720);
    BSP_MotorControl_SetMaxSpeed(2, maxSpeed_tilt);
    BSP_MotorControl_SetAcceleration(2, 4000);
    BSP_MotorControl_SetDeceleration(2, 4000);

    last_msg_time = HAL_GetTick();

    // Start listening for UART data
    if (HAL_UART_Receive_IT(&huart2, &rx_byte, 1) != HAL_OK) {
        Error_Handler(0);
    }

    while (1)
    {
        // --- ISR-SAFE DOUBLE BUFFER PARSING ---
        if (string_ready)
        {
            __disable_irq();
            string_ready = 0;
            uint8_t read_idx = 1 - write_idx; // Read from the buffer the ISR is NOT using
            __enable_irq();

            float temp_phi, temp_theta;

            // Strict Validation (Requires exactly two valid floats separated by a comma)
            if (sscanf(rx_buf[read_idx], "P:%f,T:%f", &temp_phi, &temp_theta) == 2)
            {
            	int32_t raw_pan_steps = BSP_MotorControl_GetPosition(0);
            	float a = (float)raw_pan_steps * RAD_PER_STEP_PAN; //convert to radians
            	a = fmodf(a, 2.0f * M_PI);
            	if (a > M_PI)  a -= 2.0f * M_PI;
            	if (a < -M_PI) a += 2.0f * M_PI;

            	float at_raw = temp_phi + a;
            	float diff_a = at_raw - target_alpha;
            	float diff_b = temp_theta - target_beta; //bt_raw = temp_beta
            	if (diff_a > M_PI) diff_a -= 2.0f * M_PI;
            	if (diff_a < -M_PI) diff_a += 2.0f * M_PI;
            	if (fabs(diff_a) > DEADBAND_RAD){
            		target_alpha = at_raw;
            	}

            	if (fabs(diff_b) > DEADBAND_RAD){
            		target_beta = temp_theta;
            	}

                last_msg_time = HAL_GetTick(); // Reset heartbeat
            }
        }

        // Continuously process motor movements toward target
        Process_Motor_Control();

        // Small delay to prevent tight loop lockup
        HAL_Delay(5);
    }
}

/* --- Motor Processing Logic (PID Controlled) --- */
void Process_Motor_Control(void)
{
    // Static variables preserve their state between function calls
    static float prev_error_a = 0.0f;
    static float prev_error_b = 0.0f;
    static float error_int_a = 0.0f;
    static float error_int_b = 0.0f;
    static uint32_t prev_time = 0;
    static int32_t prev_pan_steps_per_sec = 0;
    static int32_t prev_tilt_steps_per_sec = 0;
    const int32_t SPEED_THRESHOLD = 1;
    const float error_int_limit_pan = 3.0;
    const float error_int_limit_tilt = 0.5;
    const float TRACKING_JUMP_THRESHOLD = 5.0f * (M_PI / 180.0f);
    const float derivative_limit = 2.0f;
    const float derivative_limit_brake_pan = 2.0f;
    const float derivative_limit_brake_tilt = 0.6f;

    // Track targets to detect when a new command arrives
	static float last_target_alpha = 0.0f;
	static float last_target_beta = 0.0f;

    // 1. Get current physical step positions
    int32_t raw_pan_steps = BSP_MotorControl_GetPosition(0);
    int32_t raw_tilt_steps = BSP_MotorControl_GetPosition(2);

    // 2. Convert to radians
    float a_current = (float)raw_pan_steps * RAD_PER_STEP_PAN;
    float b_raw_current = (float)raw_tilt_steps * RAD_PER_STEP_TILT;
    float b_current = -b_raw_current;

    // 3. The infinite unwind wrap-around fix for Pan
    a_current = fmodf(a_current, 2.0f * M_PI);
    if (a_current > M_PI)  a_current -= 2.0f * M_PI;
    if (a_current < -M_PI) a_current += 2.0f * M_PI;

    // 4. Stop if python disconnects (1500ms timeout)
    uint32_t curr_time_ms = HAL_GetTick();
    if ((curr_time_ms - last_msg_time) > 1500) {
        target_alpha = a_current;
        target_beta = b_current;
        BSP_MotorControl_SoftStop(0);
        BSP_MotorControl_SoftStop(2);

        // Reset PID states so it doesn't jerk when reconnected
        error_int_a = 0;
        error_int_b = 0;
        prev_time = curr_time_ms;
        return;
    }

    // Reset Integral ONLY on major angle changes (Jumps)
	if (target_alpha != last_target_alpha) {
		if (fabs(target_alpha - last_target_alpha) > TRACKING_JUMP_THRESHOLD) {
			error_int_a = 0.0f; // Wipe memory to prevent massive transition windup
		}
		last_target_alpha = target_alpha; // Always update the tracking anchor
	}

	if (target_beta != last_target_beta) {
		if (fabs(target_beta - last_target_beta) > TRACKING_JUMP_THRESHOLD) {
			error_int_b = 0.0f;
		}
		last_target_beta = target_beta;
	}

    // 5. Calculate Time Delta (dt) in seconds safely
    float dt = (float)(curr_time_ms - prev_time) / 1000.0f;
    if (dt <= 0.0f) dt = 0.001f; // Mathematically prevent divide-by-zero

    // 6. Calculate shortest-path errors
    float error_a = target_alpha - a_current;
    float error_b = target_beta - b_current;

    if (error_a > M_PI)  error_a -= 2.0f * M_PI;
    if (error_a < -M_PI) error_a += 2.0f * M_PI;

    const float ERROR_DEADBAND = 0.1f * (M_PI / 180.0f);

    if ((error_a > 0 && prev_error_a < 0) || (error_a < 0 && prev_error_a > 0)) {
		error_int_a = 0.0f;
	}
	if ((error_b > 0 && prev_error_b < 0) || (error_b < 0 && prev_error_b > 0)) {
		error_int_b = 0.0f;
	}

    if (fabs(error_a) < ERROR_DEADBAND) { // ~0.45 degrees
		error_int_a = 0.0f;       // Clear it so it doesn't hum/hunt while sitting still
	} else if (fabs(error_a) < 100 * ERROR_DEADBAND){
		error_int_a += 3 * error_a * dt;
	} else if (fabs(error_a) < 150 * ERROR_DEADBAND) { // Only integrate if error is less than 10 degrees
		error_int_a += error_a * dt;
	} //else {
		//error_int_a *= 0.95f;     // Leaky integral: actively decay memory during large transitions
	//}

    if (fabs(error_b) < ERROR_DEADBAND) { // ~0.45 degrees
		error_int_b = 0.0f; // Clear it so it doesn't hum/hunt while sitting still
    } else if (fabs(error_b) < 100 * ERROR_DEADBAND){
    	error_int_b += 3 * error_b * dt;
	} else if (fabs(error_b) < 150 * ERROR_DEADBAND) { // Only integrate if error is less than 10 degrees
		error_int_b += error_b * dt;
	}

    /*
    // 7. PID Math
    error_int_a += error_a * dt;
    error_int_b += error_b * dt;
    */

    // Anti-Windup: Clamp the integral so it doesn't accumulate to infinity
    // if the motors are physically blocked or catching up.
    if (error_int_a > error_int_limit_pan) error_int_a = error_int_limit_pan;
    if (error_int_a < -error_int_limit_pan) error_int_a = -error_int_limit_pan;
    if (error_int_b > error_int_limit_tilt) error_int_b = error_int_limit_tilt;
    if (error_int_b < -error_int_limit_tilt) error_int_b = -error_int_limit_tilt;

    float de_dt_a = (error_a - prev_error_a) / dt;
    float de_dt_b = (error_b - prev_error_b) / dt;

    if ((error_a > 0 && de_dt_a < 0) || (error_a < 0 && de_dt_a > 0)) {
    	if (de_dt_a > derivative_limit_brake_pan) de_dt_a = derivative_limit_brake_pan;
    	if (de_dt_a < -derivative_limit_brake_pan) de_dt_a = -derivative_limit_brake_pan;
	}

    if ((error_b > 0 && de_dt_b < 0) || (error_b < 0 && de_dt_b > 0)) {
		if (de_dt_b > derivative_limit_brake_tilt) de_dt_b = derivative_limit_brake_tilt;
		if (de_dt_b < -derivative_limit_brake_tilt) de_dt_b = -derivative_limit_brake_tilt;
	}

    if (de_dt_a > derivative_limit) de_dt_a = derivative_limit; //clamped the derivative
	if (de_dt_a < -derivative_limit) de_dt_a = -derivative_limit;
	if (de_dt_b > derivative_limit) de_dt_b = derivative_limit;
	if (de_dt_b < -derivative_limit) de_dt_b = -derivative_limit;


    float speed_a_rad = (K_pa * error_a) + (K_ia * error_int_a) + (K_da * de_dt_a);
    float speed_b_rad = (K_pb * error_b) + (K_ib * error_int_b) + (K_db * de_dt_b);

    // Update memory for next loop
    prev_error_a = error_a;
    prev_error_b = error_b;
    prev_time = curr_time_ms;

    // 8. Convert Math Output to Physical Steps/Sec
    int32_t desired_pan_steps_per_sec = (int32_t)(speed_a_rad / RAD_PER_STEP_PAN);

    // Remember to invert the tilt command to match the physical gearing!
    int32_t desired_tilt_steps_per_sec = (int32_t)(-speed_b_rad / RAD_PER_STEP_TILT);

    const int32_t MAX_VELOCITY_STEP_CHANGE = 1000;

	int32_t pan_steps_per_sec;
	int32_t diff_pan = desired_pan_steps_per_sec - prev_pan_steps_per_sec;
	if (diff_pan > MAX_VELOCITY_STEP_CHANGE)       pan_steps_per_sec = prev_pan_steps_per_sec + MAX_VELOCITY_STEP_CHANGE;
	else if (diff_pan < -MAX_VELOCITY_STEP_CHANGE) pan_steps_per_sec = prev_pan_steps_per_sec - MAX_VELOCITY_STEP_CHANGE;
	else                                           pan_steps_per_sec = desired_pan_steps_per_sec;

	int32_t tilt_steps_per_sec;
	int32_t diff_tilt = desired_tilt_steps_per_sec - prev_tilt_steps_per_sec;
	if (diff_tilt > MAX_VELOCITY_STEP_CHANGE)       tilt_steps_per_sec = prev_tilt_steps_per_sec + MAX_VELOCITY_STEP_CHANGE;
	else if (diff_tilt < -MAX_VELOCITY_STEP_CHANGE) tilt_steps_per_sec = prev_tilt_steps_per_sec - MAX_VELOCITY_STEP_CHANGE;
	else                                           tilt_steps_per_sec = desired_tilt_steps_per_sec;

    // Extract pure magnitude (absolute value) for the driver constraints
    uint16_t pan_speed_abs = (uint16_t)abs(pan_steps_per_sec);
    uint16_t tilt_speed_abs = (uint16_t)abs(tilt_steps_per_sec);

    // Hardware Safety Clamps
    if (pan_speed_abs > maxSpeed_pan) pan_speed_abs = maxSpeed_pan;
    if (tilt_speed_abs > maxSpeed_tilt) tilt_speed_abs = maxSpeed_tilt;

    // 9. Act on Pan Motor
    int pan_dir = (pan_steps_per_sec > 0) ? FORWARD : BACKWARD;
    int prev_pan_dir = (prev_pan_steps_per_sec > 0) ? FORWARD : BACKWARD;
    if (pan_speed_abs < 1) {
    	if (prev_pan_steps_per_sec != 0) { // Only spam the SPI bus once per stop
			BSP_MotorControl_SoftStop(0);
			prev_pan_steps_per_sec = 0;    // Clear the ghost speed!
		}
    } else if (abs(pan_steps_per_sec - prev_pan_steps_per_sec) > SPEED_THRESHOLD) {
        BSP_MotorControl_SetMaxSpeed(0, pan_speed_abs);
        if (prev_pan_steps_per_sec == 0 || pan_dir != prev_pan_dir) {
				BSP_MotorControl_Run(0, pan_dir);
			}
        //BSP_MotorControl_Run(0, (pan_steps_per_sec > 0) ? FORWARD : BACKWARD);
        prev_pan_steps_per_sec = pan_steps_per_sec;
    }

    // 10. Act on Tilt Motor
    int tilt_dir = (tilt_steps_per_sec > 0) ? FORWARD : BACKWARD;
    int prev_tilt_dir = (prev_tilt_steps_per_sec > 0) ? FORWARD : BACKWARD;
    if (tilt_speed_abs < 1) {
    	if (prev_tilt_steps_per_sec != 0) {
    		BSP_MotorControl_SoftStop(2);
    		prev_tilt_steps_per_sec = 0;
    	}
    } else if (abs(tilt_steps_per_sec - prev_tilt_steps_per_sec) > SPEED_THRESHOLD) {
        BSP_MotorControl_SetMaxSpeed(2, tilt_speed_abs);
        if (prev_tilt_steps_per_sec == 0 || tilt_dir != prev_tilt_dir) {
				BSP_MotorControl_Run(2, tilt_dir);
			}
        //BSP_MotorControl_Run(2, (tilt_steps_per_sec > 0) ? FORWARD : BACKWARD);
        prev_tilt_steps_per_sec = tilt_steps_per_sec;
    }
}

/* --- UART Interrupt Callback (Double-Buffered, Race-Free, with Overflow Recovery) --- */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    // Static flag to track if we are currently discarding a corrupted/oversized stream
    static uint8_t overflow_flag = 0;

    if (rx_byte == '\n') {
        if (overflow_flag) {
            // We finally found a newline, but the message before it was oversized/corrupted.
            // Discard the garbage, clear the flag, and silently resynchronize.
            overflow_flag = 0;
            rx_index = 0;
        } else {
            // 1. Terminate the valid string
            rx_buf[write_idx][rx_index] = '\0';

            // 2. EXPLICIT OWNERSHIP HANDOFF: Flip the index FIRST
            write_idx = 1 - write_idx;
            rx_index = 0;

            // 3. Raise the flag LAST. The main loop will now read (1 - write_idx),
            // which safely points to the buffer we just finished filling.
            string_ready = 1;
        }
    } else {
        if (rx_index < 31) {
            // Normal operation: store byte
            rx_buf[write_idx][rx_index++] = rx_byte;
        } else {
            // Memory safety hit: The string is too long. Stop recording and enter discard mode.
            overflow_flag = 1;
        }
    }

    // Safely re-arm with checked return value
    if (HAL_UART_Receive_IT(huart, &rx_byte, 1) != HAL_OK) {
        // Force state machine reset if hardware locks up during re-arm
        huart->RxState = HAL_UART_STATE_READY;
        HAL_UART_Receive_IT(huart, &rx_byte, 1);
    }
}

/* --- UART Error Callback (Catches Overruns and Prevents Freezing) --- */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        __HAL_UART_CLEAR_OREFLAG(huart);
        __HAL_UART_CLEAR_NEFLAG(huart);
        __HAL_UART_CLEAR_FEFLAG(huart);

        // Hard reset Rx state and buffer index to prevent trailing garbage
        rx_index = 0;
        huart->RxState = HAL_UART_STATE_READY;
        HAL_UART_Receive_IT(huart, &rx_byte, 1);
    }
}

/* --- Required BSP Interrupt/Error Handlers --- */
void MyFlagInterruptHandler(void)
{
    BSP_MotorControl_CmdGetStatus(0);
    BSP_MotorControl_CmdGetStatus(1);
    BSP_MotorControl_CmdGetStatus(2);
}

void Error_Handler(uint16_t error)
{
    while(1) { } // Trap errors
}

/* --- Hardware Initialization Functions --- */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK) {
      Error_Handler(0);
  }
}

void HAL_UART_MspInit(UART_HandleTypeDef* huart)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(huart->Instance == USART2)
  {
    __HAL_RCC_USART2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* This turns the interrupt on! */
    HAL_NVIC_SetPriority(USART2_IRQn, 0, 1);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
  }
}

void HAL_UART_MspDeInit(UART_HandleTypeDef* huart)
{
  if(huart->Instance == USART2)
  {
    __HAL_RCC_USART2_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_2|GPIO_PIN_3);
    HAL_NVIC_DisableIRQ(USART2_IRQn);
  }
}

/* Provide empty system clock config to satisfy linker if not generated */
__weak void SystemClock_Config(void) {}

/* --- UART/DMA STUBS (Prevents Linker Errors) --- */
HAL_StatusTypeDef HAL_DMA_Abort(DMA_HandleTypeDef *hdma) { return HAL_OK; }
HAL_StatusTypeDef HAL_DMA_Abort_IT(DMA_HandleTypeDef *hdma) { return HAL_OK; }

/* --- SYSTEM CALL STUBS (Prevents libc_nano errors) --- */
#include <sys/stat.h>
__attribute__((weak)) int _close(int file) { return -1; }
__attribute__((weak)) int _fstat(int file, struct stat *st) { st->st_mode = S_IFCHR; return 0; }
__attribute__((weak)) int _isatty(int file) { return 1; }
__attribute__((weak)) int _lseek(int file, int ptr, int dir) { return 0; }
__attribute__((weak)) int _read(int file, char *ptr, int len) { return 0; }
__attribute__((weak)) int _write(int file, char *ptr, int len) { return len; }
__attribute__((weak)) int _kill(int pid, int sig) { return -1; }
__attribute__((weak)) int _getpid(void) { return 1; }

__attribute__((weak)) void *_sbrk(int incr) {
    extern char _end;
    static char *heap_end = 0;
    char *prev_heap_end;
    if (heap_end == 0) heap_end = &_end;
    prev_heap_end = heap_end;
    heap_end += incr;
    return (void *)prev_heap_end;
}
