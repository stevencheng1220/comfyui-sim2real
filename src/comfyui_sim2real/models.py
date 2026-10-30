# ComfyUI reads INPUT_TYPES as {"required": {socket name: (type name,) or (type name, options)}}.
type InputSocket = tuple[str] | tuple[str, dict[str, object]]
type InputTypes = dict[str, dict[str, InputSocket]]
