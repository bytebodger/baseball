import { createEndpoints } from './common/functions/createEndpoints.js';
import { initialize } from './common/functions/initialize.js';
import { scrape } from './common/functions/scrape.js';
createEndpoints(initialize());
scrape();
