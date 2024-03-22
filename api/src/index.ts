import { createEndpoints } from './common/functions/createEndpoints.js';
import { initialize } from './common/functions/initialize.js';
import { runScrapers } from './common/functions/runScrapers.js';

createEndpoints(await initialize());
await runScrapers();