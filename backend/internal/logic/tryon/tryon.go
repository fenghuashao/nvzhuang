package tryon

import (
	"context"
	"encoding/base64"
	"strings"

	"github.com/gogf/gf/v2/errors/gerror"

	"backend/internal/imagevalidate"
)

// DoTryon 执行换装：选择模型 → 构建提示词 → 调用生成 → 返回结果
func DoTryon(ctx context.Context, personB64, garmentB64, model, size, lang string) (resultB64, usedModel, usedPrompt string, err error) {
	return DoTryonWithClientConfig(ctx, personB64, garmentB64, model, size, lang, nil)
}

// DoTryonWithClientConfig 支持请求级临时模型配置。
// clientConfig 不会进入全局配置、缓存、数据库或日志。
func DoTryonWithClientConfig(ctx context.Context, personB64, garmentB64, model, size, lang string, clientConfig *ModelConfig) (resultB64, usedModel, usedPrompt string, err error) {
	// 参数校验
	if personB64 == "" {
		return "", "", "", gerror.New("人像照片不能为空")
	}
	if garmentB64 == "" {
		return "", "", "", gerror.New("服装图不能为空")
	}
	if err := decodeB64(personB64); err != nil {
		return "", "", "", gerror.New("人像照片 base64 格式错误")
	}
	if err := decodeB64(garmentB64); err != nil {
		return "", "", "", gerror.New("服装图 base64 格式错误")
	}

	// 默认模型：Gemini（Nano Banana 2），因为 GPT Image 系列对真人换装场景
	// 会被官方安全审核拦截（moderation_blocked / sexual），Gemini 无此限制。
	if model == "" {
		model = "gemini"
	}

	var gen ImageGenerator
	if clientConfig != nil {
		gen, err = NewGeneratorFromClientConfig(clientConfig)
	} else {
		gen, err = NewGenerator(ctx, model)
	}
	if err != nil {
		return "", "", "", err
	}

	// 构建提示词：
	// 1. GPT-Image 系列专门使用防封控安全提示词；
	// 2. Gemini 保留原英文稳定提示词；
	// 3. 豆包 / MiniMax 等其他模型保留原提示词（按语言选择）。
	//    MiniMax 走「人像 subject_reference + 纯服装描述」，最终提示词由适配器内部拼装。
	var promptStr string
	isGemini := (clientConfig != nil && strings.EqualFold(clientConfig.Provider, "gemini")) || gen.Name() == "gemini"
	if isGPTImageModel(model, clientConfig, gen.Name()) {
		promptStr = BuildGPTSafety(lang)
	} else if isGemini {
		promptStr = BuildGemini(lang)
	} else {
		// 豆包 / MiniMax 等其他模型使用通用提示词。
		// MiniMax 适配器内部会再拼装「身份描述 + 服装描述」后的最终提示词。
		promptStr = BuildDefault(lang)
	}

	// 调用生成
	result, err := gen.Generate(ctx, personB64, garmentB64, promptStr, size)
	if err != nil {
		// 不记录原始错误：网络错误可能包含用户自定义 Base URL，供应商响应也可能含敏感信息。
		return "", "", "", gerror.Wrap(err, "换装生成失败，请稍后重试")
	}

	return result, gen.Name(), resolveUsedPrompt(gen, promptStr), nil
}

// resolveUsedPrompt 返回实际发给模型的提示词。
// 部分适配器（如 MiniMax）会在生成过程中自行拼装最终提示词，
// 若直接返回构建阶段的模板，接口返回的 prompt 与真实提示词不符，排查时极具误导性。
func resolveUsedPrompt(gen ImageGenerator, fallback string) string {
	if reporter, ok := gen.(PromptReporter); ok {
		if used := reporter.UsedPrompt(); used != "" {
			return used
		}
	}
	return fallback
}

// DoTryonWithValidatedImages 使用已完成安全校验的图片结构执行换装
func DoTryonWithValidatedImages(ctx context.Context, personImg, garmentImg *imagevalidate.ValidatedImage, model, size, lang string, clientConfig *ModelConfig) (resultB64, usedModel, usedPrompt string, err error) {
	resultB64, usedModel, usedPrompt, _, err = DoTryonWithValidatedImagesMode(ctx, personImg, garmentImg, model, size, lang, clientConfig)
	return resultB64, usedModel, usedPrompt, err
}

// DoTryonWithValidatedImagesMode 与 DoTryonWithValidatedImages 相同，
// 额外返回本次实际走的生成路径（mode）。
// mode 用于让调用方知道结果是否走了降级路径 —— 例如 MiniMax 在参考图
// 被内容审核拦下时会退回纯文生图，此时人脸保真度较低，需要如实告知用户。
func DoTryonWithValidatedImagesMode(ctx context.Context, personImg, garmentImg *imagevalidate.ValidatedImage, model, size, lang string, clientConfig *ModelConfig) (resultB64, usedModel, usedPrompt, usedMode string, err error) {
	if personImg == nil || len(personImg.Data) == 0 {
		return "", "", "", "", gerror.New("人像照片不能为空")
	}
	if garmentImg == nil || len(garmentImg.Data) == 0 {
		return "", "", "", "", gerror.New("服装图不能为空")
	}

	personB64 := base64.StdEncoding.EncodeToString(personImg.Data)
	garmentB64 := base64.StdEncoding.EncodeToString(garmentImg.Data)

	if model == "" {
		model = "gemini"
	}

	var gen ImageGenerator
	if clientConfig != nil {
		gen, err = NewGeneratorFromClientConfig(clientConfig)
	} else {
		gen, err = NewGenerator(ctx, model)
	}
	if err != nil {
		return "", "", "", "", err
	}

	var promptStr string
	isGemini := (clientConfig != nil && strings.EqualFold(clientConfig.Provider, "gemini")) || gen.Name() == "gemini"
	if isGPTImageModel(model, clientConfig, gen.Name()) {
		promptStr = BuildGPTSafety(lang)
	} else if isGemini {
		promptStr = BuildGemini(lang)
	} else {
		promptStr = BuildDefault(lang)
	}

	result, err := gen.Generate(ctx, personB64, garmentB64, promptStr, size)
	if err != nil {
		return "", "", "", "", gerror.Wrap(err, "换装生成失败，请稍后重试")
	}

	return result, gen.Name(), resolveUsedPrompt(gen, promptStr), resolveUsedMode(gen), nil
}

// resolveUsedMode 返回实际走的生成路径。未实现 ModeReporter 的适配器返回空串。
func resolveUsedMode(gen ImageGenerator) string {
	if reporter, ok := gen.(ModeReporter); ok {
		return reporter.Mode()
	}
	return ""
}

// isGPTImageModel 判断当前请求是否使用 GPT-Image 系列模型
func isGPTImageModel(model string, clientConfig *ModelConfig, genName string) bool {
	check := func(s string) bool {
		lower := strings.ToLower(s)
		return strings.Contains(lower, "gpt-image") || strings.Contains(lower, "gpt_image")
	}
	if check(genName) || check(model) {
		return true
	}
	if clientConfig != nil && (check(clientConfig.Model) || check(clientConfig.Name)) {
		return true
	}
	return false
}
